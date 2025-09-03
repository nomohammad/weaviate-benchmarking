package cmd

import (
	"context"
	"crypto/tls"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-openapi/strfmt"
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/retry"
	"github.com/hashicorp/go-retryablehttp"
	log "github.com/sirupsen/logrus"
	"golang.org/x/exp/constraints"

	"github.com/google/uuid"
	"github.com/spf13/cobra"
	"github.com/weaviate/hdf5"
	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/auth"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/fault"
	"github.com/weaviate/weaviate/entities/models"
	weaviategrpc "github.com/weaviate/weaviate/grpc/generated/protocol/v1"
	"github.com/weaviate/weaviate/usecases/byteops"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/structpb"
)

type CompressionType byte

const (
	CompressionTypePQ CompressionType = 0
	CompressionTypeSQ CompressionType = 1
	CompressionTypeRQ CompressionType = 2
)

// Batch of vectors and offset for writing to Weaviate
type Batch struct {
	Vectors [][]float32
	Offset  int
	Filters []int
}

// Weaviate https://github.com/weaviate/weaviate-chaos-engineering/tree/main/apps/ann-benchmarks style format
// mixed camel / snake case for compatibility
type ResultsJSONBenchmark struct {
	Api              string  `json:"api"`
	Ef               int     `json:"ef"`
	EfConstruction   int     `json:"efConstruction"`
	MaxConnections   int     `json:"maxConnections"`
	Mean             float64 `json:"meanLatency"`
	P99Latency       float64 `json:"p99Latency"`
	QueriesPerSecond float64 `json:"qps"`
	Shards           int     `json:"shards"`
	Parallelization  int     `json:"parallelization"`
	Limit            int     `json:"limit"`
	ImportTime       float64 `json:"importTime"`
	RunID            string  `json:"run_id"`
	Dataset          string  `json:"dataset_file"`
	Recall           float64 `json:"recall"`
	NDCG             float64 `json:"ndcg"`
	HeapAllocBytes   float64 `json:"heap_alloc_bytes"`
	HeapInuseBytes   float64 `json:"heap_inuse_bytes"`
	HeapSysBytes     float64 `json:"heap_sys_bytes"`
	Timestamp        string  `json:"timestamp"`
}

// Convert an int to a uuid formatted string
func uuidFromInt(val int) string {
	bytes := make([]byte, 16)
	binary.BigEndian.PutUint64(bytes[8:], uint64(val))
	id, err := uuid.FromBytes(bytes)
	if err != nil {
		panic(err)
	}

	return id.String()
}

// Convert a uuid formatted string to an int
func intFromUUID(uuidStr string) int {
	id, err := uuid.Parse(uuidStr)
	if err != nil {
		panic(err)
	}
	val := binary.BigEndian.Uint64(id[8:])
	return int(val)
}

// Writes a single batch of vectors to Weaviate using gRPC
func writeChunk(chunk *Batch, client *weaviategrpc.WeaviateClient, cfg *Config) {
	objects := make([]*weaviategrpc.BatchObject, len(chunk.Vectors))

	for i, vector := range chunk.Vectors {
		objects[i] = &weaviategrpc.BatchObject{
			Uuid:       uuidFromInt(i + chunk.Offset + cfg.Offset),
			Collection: cfg.ClassName,
		}
		if cfg.Tenant != "" {
			objects[i].Tenant = cfg.Tenant
		}
		if cfg.MultiVectorDimensions > 0 {
			if len(vector)%cfg.MultiVectorDimensions != 0 {
				log.Fatalf("Vector length %d is not a multiple of dimensions %d",
					len(vector), cfg.MultiVectorDimensions)
			}
			rows := len(vector) / cfg.MultiVectorDimensions

			multiVec := make([][]float32, rows)
			for i := 0; i < rows; i++ {
				start := i * cfg.MultiVectorDimensions
				end := start + cfg.MultiVectorDimensions
				multiVec[i] = vector[start:end]
			}
			objects[i].Vectors = []*weaviategrpc.Vectors{{
				Name:        "multivector",
				VectorBytes: byteops.Fp32SliceOfSlicesToBytes(multiVec),
				Type:        weaviategrpc.Vectors_VECTOR_TYPE_MULTI_FP32,
			}}
		} else {
			objects[i].VectorBytes = encodeVector(vector)
		}
		if cfg.NamedVector != "" {
			vectors := make([]*weaviategrpc.Vectors, 1)
			vectors[0] = &weaviategrpc.Vectors{
				VectorBytes: encodeVector(vector),
				Name:        cfg.NamedVector,
			}
			objects[i].Vectors = vectors
		}
		if cfg.Filter {
			nonRefProperties, err := structpb.NewStruct(map[string]interface{}{
				"category": strconv.Itoa(chunk.Filters[i]),
			})
			if err != nil {
				log.Fatalf("Error creating filtered struct: %v", err)
			}
			objects[i].Properties = &weaviategrpc.BatchObject_Properties{
				NonRefProperties: nonRefProperties,
			}
		}
	}

	batchRequest := &weaviategrpc.BatchObjectsRequest{
		Objects: objects,
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*300)
	defer cancel()

	if cfg.HttpAuth != "" {
		md := metadata.Pairs(
			"Authorization", fmt.Sprintf("Bearer %s", cfg.HttpAuth),
		)
		ctx = metadata.NewOutgoingContext(ctx, md)
	}

	response, err := (*client).BatchObjects(ctx, batchRequest)
	if err != nil {
		log.Fatalf("could not send batch: %v", err)
	}

	for _, result := range response.GetErrors() {
		if result.Error != "" {
			log.Printf("Error for index %d: %s", result.Index, result.Error)
		} else {
			log.Printf("Successfully processed object at index %d", result.Index)
		}
	}
}

func createClient(cfg *Config) *weaviate.Client {
	retryClient := retryablehttp.NewClient()
	retryClient.RetryMax = 10

	wcfg := weaviate.Config{
		Host:             cfg.HttpOrigin,
		Scheme:           cfg.HttpScheme,
		ConnectionClient: retryClient.HTTPClient,
		StartupTimeout:   60 * time.Second,
	}
	if cfg.HttpAuth != "" {
		wcfg.AuthConfig = auth.ApiKey{Value: cfg.HttpAuth}
		wcfg.ConnectionClient = nil
	}
	client, err := weaviate.NewClient(wcfg)
	if err != nil {
		log.Fatalf("Error creating client: %v", err)
	}
	return client
}

// Re/create Weaviate schema
func createSchema(cfg *Config, client *weaviate.Client) {
	err := client.Schema().ClassDeleter().WithClassName(cfg.ClassName).Do(context.Background())
	if err != nil {
		log.Fatalf("Error deleting class: %v", err)
	}

	multiTenancyEnabled := false
	if cfg.NumTenants > 0 {
		multiTenancyEnabled = true
	}

	var classObj = &models.Class{
		Class:       cfg.ClassName,
		Description: fmt.Sprintf("Created by the Weaviate Benchmarker at %s", time.Now().String()),
		MultiTenancyConfig: &models.MultiTenancyConfig{
			Enabled: multiTenancyEnabled,
		},
	}

	if cfg.Shards > 1 {
		classObj.ShardingConfig = map[string]interface{}{
			"desiredCount": cfg.Shards,
		}
	}

	var vectorIndexConfig map[string]interface{}

	if cfg.IndexType == "hnsw" {
		vectorIndexConfig = map[string]interface{}{
			"distance":               cfg.DistanceMetric,
			"efConstruction":         float64(cfg.EfConstruction),
			"maxConnections":         float64(cfg.MaxConnections),
			"cleanupIntervalSeconds": cfg.CleanupIntervalSeconds,
			"flatSearchCutoff":       cfg.FlatSearchCutoff,
		}
		if cfg.PQ == "auto" {
			pqConfig := map[string]interface{}{
				"enabled":       true,
				"segments":      cfg.PQSegments,
				"trainingLimit": cfg.TrainingLimit,
			}
			if cfg.RescoreLimit > -1 {
				pqConfig["rescoreLimit"] = cfg.RescoreLimit
			}
			vectorIndexConfig["pq"] = pqConfig
		} else if cfg.BQ {
			bqConfig := map[string]interface{}{
				"enabled": true,
			}
			if cfg.RescoreLimit > -1 {
				bqConfig["rescoreLimit"] = cfg.RescoreLimit
			}
			vectorIndexConfig["bq"] = bqConfig
		} else if cfg.SQ == "auto" {
			vectorIndexConfig = map[string]interface{}{
				"distance":               cfg.DistanceMetric,
				"efConstruction":         float64(cfg.EfConstruction),
				"maxConnections":         float64(cfg.MaxConnections),
				"cleanupIntervalSeconds": cfg.CleanupIntervalSeconds,
				"sq": map[string]interface{}{
					"enabled":       true,
					"trainingLimit": cfg.TrainingLimit,
				},
			}
		} else if cfg.RQ == "auto" {
			rqConfig := map[string]interface{}{
				"enabled": true,
				"bits":    cfg.RQBits,
			}
			if cfg.RescoreLimit > -1 {
				rqConfig["rescoreLimit"] = cfg.RescoreLimit
			}
			vectorIndexConfig = map[string]interface{}{
				"distance":               cfg.DistanceMetric,
				"efConstruction":         float64(cfg.EfConstruction),
				"maxConnections":         float64(cfg.MaxConnections),
				"cleanupIntervalSeconds": cfg.CleanupIntervalSeconds,
				"rq":                     rqConfig,
			}
		}
	} else if cfg.IndexType == "flat" {
		vectorIndexConfig = map[string]interface{}{
			"distance": cfg.DistanceMetric,
		}
		if cfg.BQ {
			bqConfig := map[string]interface{}{
				"enabled": true,
				"cache":   cfg.Cache,
			}
			if cfg.RescoreLimit > -1 {
				bqConfig["rescoreLimit"] = cfg.RescoreLimit
			}
			vectorIndexConfig["bq"] = bqConfig
		}
	} else if cfg.IndexType == "dynamic" {
		log.WithFields(log.Fields{"threshold": cfg.DynamicThreshold}).Info("Building dynamic vector index")
		vectorIndexConfig = map[string]interface{}{
			"distance":  cfg.DistanceMetric,
			"threshold": cfg.DynamicThreshold,
			"hnsw": map[string]interface{}{
				"efConstruction":         float64(cfg.EfConstruction),
				"maxConnections":         float64(cfg.MaxConnections),
				"cleanupIntervalSeconds": cfg.CleanupIntervalSeconds,
				"flatSearchCutoff":       cfg.FlatSearchCutoff,
			},
		}
		if cfg.PQ == "auto" {
			pqConfig := map[string]interface{}{
				"enabled":       true,
				"segments":      cfg.PQSegments,
				"trainingLimit": cfg.TrainingLimit,
			}
			if cfg.RescoreLimit > -1 {
				pqConfig["rescoreLimit"] = cfg.RescoreLimit
			}
			vectorIndexConfig["hnsw"].(map[string]interface{})["pq"] = pqConfig
		} else if cfg.BQ {
			bqConfig := map[string]interface{}{
				"enabled": true,
				"cache":   true,
			}
			if cfg.RescoreLimit > -1 {
				bqConfig["rescoreLimit"] = cfg.RescoreLimit
			}
			vectorIndexConfig["hnsw"].(map[string]interface{})["bq"] = bqConfig
		}
	} else {
		log.Fatalf("Unknown index type %s", cfg.IndexType)
	}

	vectorIndexConfig["filterStrategy"] = cfg.FilterStrategy

	if cfg.NamedVector != "" {
		vectorConfig := make(map[string]models.VectorConfig)
		vectorConfig[cfg.NamedVector] = models.VectorConfig{
			Vectorizer:        map[string]interface{}{"none": nil},
			VectorIndexType:   cfg.IndexType,
			VectorIndexConfig: vectorIndexConfig,
		}
		classObj.VectorConfig = vectorConfig
	} else {
		if cfg.MultiVectorDimensions > 0 {
			vectorIndexConfig = map[string]interface{}{}
			if cfg.PQ == "auto" {
				pqConfig := map[string]interface{}{
					"enabled":       true,
					"segments":      cfg.PQSegments,
					"trainingLimit": cfg.TrainingLimit,
				}
				if cfg.RescoreLimit > -1 {
					pqConfig["rescoreLimit"] = cfg.RescoreLimit
				}
				vectorIndexConfig["pq"] = pqConfig
			} else if cfg.BQ {
				bqConfig := map[string]interface{}{
					"enabled": true,
					"cache":   true,
				}
				if cfg.RescoreLimit > -1 {
					bqConfig["rescoreLimit"] = cfg.RescoreLimit
				}
				vectorIndexConfig["bq"] = bqConfig
			} else if cfg.SQ == "auto" {
				vectorIndexConfig = map[string]interface{}{
					"distance":               cfg.DistanceMetric,
					"efConstruction":         float64(cfg.EfConstruction),
					"maxConnections":         float64(cfg.MaxConnections),
					"cleanupIntervalSeconds": cfg.CleanupIntervalSeconds,
					"sq": map[string]interface{}{
						"enabled":       true,
						"trainingLimit": cfg.TrainingLimit,
					},
				}
			} else if cfg.RQ == "auto" {
				rqConfig := map[string]interface{}{
					"enabled": true,
					"bits":    cfg.RQBits,
				}
				if cfg.RescoreLimit > -1 {
					rqConfig["rescoreLimit"] = cfg.RescoreLimit
				}
				vectorIndexConfig = map[string]interface{}{
					"distance":               cfg.DistanceMetric,
					"efConstruction":         float64(cfg.EfConstruction),
					"maxConnections":         float64(cfg.MaxConnections),
					"cleanupIntervalSeconds": cfg.CleanupIntervalSeconds,
					"rq":                     rqConfig,
				}
			}
			vectorIndexConfig["multivector"] = map[string]interface{}{
				"enabled": true,
				"muvera": map[string]interface{}{
					"enabled":      cfg.MuveraEnabled,
					"ksim":         cfg.MuveraKSim,
					"dprojections": cfg.MuveraDProjections,
					"repetition":   cfg.MuveraRepetition,
				},
			}

			classObj.VectorConfig = map[string]models.VectorConfig{
				"multivector": {
					Vectorizer: map[string]interface{}{
						"none": map[string]interface{}{},
					},
					VectorIndexConfig: vectorIndexConfig,
					VectorIndexType:   cfg.IndexType,
				},
			}
		} else {
			classObj.VectorIndexType = cfg.IndexType
			classObj.VectorIndexConfig = vectorIndexConfig
		}
	}

	if cfg.ReplicationFactor > 1 || cfg.AsyncReplicationEnabled {
		classObj.ReplicationConfig = &models.ReplicationConfig{
			Factor:       int64(cfg.ReplicationFactor),
			AsyncEnabled: cfg.AsyncReplicationEnabled,
		}
	}

	err = client.Schema().ClassCreator().WithClass(classObj).Do(context.Background())
	if err != nil {
		panic(err)
	}
	log.Printf("Created class %s", cfg.ClassName)
}

func deleteChunk(chunk *Batch, client *weaviate.Client, cfg *Config) {
	log.Debugf("Deleting chunk of %d vectors index %d", len(chunk.Vectors), chunk.Offset)
	for i := range chunk.Vectors {
		uuid := uuidFromInt(i + chunk.Offset + cfg.Offset)
		err := client.Data().Deleter().WithClassName(cfg.ClassName).WithID(uuid).Do(context.Background())
		if err != nil {
			log.Fatalf("Error deleting object: %v", err)
		}
	}
}

func deleteUuidSlice(cfg *Config, client *weaviate.Client, slice []int) {
	log.WithFields(log.Fields{"length": len(slice), "class": cfg.ClassName}).Printf("Deleting objects to trigger tombstone operations")
	for _, i := range slice {
		err := client.Data().Deleter().WithClassName(cfg.ClassName).WithID(uuidFromInt(i)).Do(context.Background())
		if err != nil {
			log.Fatalf("Error deleting object: %v", err)
		}
	}
	log.WithFields(log.Fields{"length": len(slice), "class": cfg.ClassName}).Printf("Completed deletes")
}

func deleteUuidRange(cfg *Config, client *weaviate.Client, start int, end int) {
	var slice []int
	for i := start; i < end; i++ {
		slice = append(slice, i)
	}
	deleteUuidSlice(cfg, client, slice)
}

func addTenantIfNeeded(cfg *Config, client *weaviate.Client) {
	if cfg.Tenant == "" {
		return
	}
	err := client.Schema().TenantsCreator().
		WithClassName(cfg.ClassName).
		WithTenants(models.Tenant{Name: cfg.Tenant}).
		Do(context.Background())
	if err != nil {
		log.Printf("Error adding tenant retrying in 1 second %v", err)
		time.Sleep(1 * time.Second)
		addTenantIfNeeded(cfg, client)
	}
}

// Update ef parameter on the Weaviate schema
func updateEf(ef int, cfg *Config, client *weaviate.Client) {
	classConfig, err := client.Schema().ClassGetter().WithClassName(cfg.ClassName).Do(context.Background())
	if err != nil {
		panic(err)
	}

	var vectorIndexConfig map[string]interface{}

	if cfg.NamedVector != "" {
		vectorIndexConfig = classConfig.VectorConfig[cfg.NamedVector].VectorIndexConfig.(map[string]interface{})
	} else if cfg.MultiVectorDimensions > 0 {
		vectorIndexConfig = classConfig.VectorConfig["multivector"].VectorIndexConfig.(map[string]interface{})
	} else {
		vectorIndexConfig = classConfig.VectorIndexConfig.(map[string]interface{})
	}

	switch cfg.IndexType {
	case "hnsw":
		vectorIndexConfig["ef"] = ef
	case "flat":
		bq := (vectorIndexConfig["bq"].(map[string]interface{}))
		bq["rescoreLimit"] = ef
	case "dynamic":
		hnswConfig := vectorIndexConfig["hnsw"].(map[string]interface{})
		hnswConfig["ef"] = ef
	}

	if cfg.NamedVector != "" {
		vectorConfig := classConfig.VectorConfig[cfg.NamedVector]
		vectorConfig.VectorIndexConfig = vectorIndexConfig
		classConfig.VectorConfig[cfg.NamedVector] = vectorConfig
	} else if cfg.MultiVectorDimensions > 0 {
		vectorConfig := classConfig.VectorConfig["multivector"]
		vectorConfig.VectorIndexConfig = vectorIndexConfig
		classConfig.VectorConfig["multivector"] = vectorConfig
	} else {
		classConfig.VectorIndexConfig = vectorIndexConfig
	}

	err = client.Schema().ClassUpdater().WithClass(classConfig).Do(context.Background())
	if err != nil {
		panic(err)
	}
}

func waitReady(cfg *Config, client *weaviate.Client, indexStart time.Time, maxDuration time.Duration, minQueueSize int64) time.Time {
	start := time.Now()
	current := time.Now()

	log.Infof("Waiting for queue to be empty\n")
	for current.Sub(start) < maxDuration {
		nodesStatus, err := client.Cluster().NodesStatusGetter().WithOutput("verbose").Do(context.Background())
		if err != nil {
			panic(err)
		}
		totalShardQueue := int64(0)
		for _, n := range nodesStatus.Nodes {
			for _, s := range n.Shards {
				if s.Class == cfg.ClassName && s.VectorQueueLength > 0 {
					totalShardQueue += s.VectorQueueLength
				}
			}
		}
		if totalShardQueue < minQueueSize {
			log.WithFields(log.Fields{"duration": current.Sub(start)}).Printf("Queue ready\n")
			log.WithFields(log.Fields{"duration": current.Sub(indexStart)}).Printf("Total load and queue ready\n")
			return current
		}
		time.Sleep(2 * time.Second)
		current = time.Now()
	}
	log.Fatalf("Queue wasn't ready in %s\n", maxDuration)
	return current
}

// Update ef parameter on the Weaviate schema
func enableCompression(cfg *Config, client *weaviate.Client, dimensions uint, compressionType CompressionType) {
	classConfig, err := client.Schema().ClassGetter().WithClassName(cfg.ClassName).Do(context.Background())
	if err != nil {
		panic(err)
	}

	var segments uint
	var vectorIndexConfig map[string]interface{}

	if cfg.MultiVectorDimensions > 0 {
		vectorIndexConfig = classConfig.VectorConfig["multivector"].VectorIndexConfig.(map[string]interface{})
	} else {
		vectorIndexConfig = classConfig.VectorIndexConfig.(map[string]interface{})
	}

	switch compressionType {
	case CompressionTypePQ:
		if dimensions%cfg.PQRatio != 0 {
			log.Fatalf("PQ ratio of %d and dimensions of %d incompatible", cfg.PQRatio, dimensions)
		}
		if !cfg.MuveraEnabled {
			segments = dimensions / cfg.PQRatio
		} else {
			segments = uint(math.Pow(2, float64(cfg.MuveraKSim))*float64(cfg.MuveraDProjections)*float64(cfg.MuveraRepetition)) / cfg.PQRatio
		}

		pqConfig := map[string]interface{}{
			"enabled":       true,
			"segments":      segments,
			"trainingLimit": cfg.TrainingLimit,
		}
		if cfg.RescoreLimit > -1 {
			pqConfig["rescoreLimit"] = cfg.RescoreLimit
		}
		vectorIndexConfig["pq"] = pqConfig
	case CompressionTypeSQ:
		sqConfig := map[string]interface{}{
			"enabled":       true,
			"trainingLimit": cfg.TrainingLimit,
		}
		if cfg.RescoreLimit > -1 {
			sqConfig["rescoreLimit"] = cfg.RescoreLimit
		}
		vectorIndexConfig["sq"] = sqConfig
	case CompressionTypeRQ:
		rqConfig := map[string]interface{}{
			"enabled": true,
			"bits":    cfg.RQBits,
		}
		if cfg.RescoreLimit > -1 {
			rqConfig["rescoreLimit"] = cfg.RescoreLimit
		}
		vectorIndexConfig["rq"] = rqConfig
	}

	if cfg.MultiVectorDimensions > 0 {
		vectorConfig := classConfig.VectorConfig["multivector"]
		vectorConfig.VectorIndexConfig = vectorIndexConfig
		classConfig.VectorConfig["multivector"] = vectorConfig
	} else {
		classConfig.VectorIndexConfig = vectorIndexConfig
	}

	err = client.Schema().ClassUpdater().WithClass(classConfig).Do(context.Background())
	if err != nil {
		panic(err)
	}
	switch compressionType {
	case CompressionTypePQ:
		log.WithFields(log.Fields{"segments": segments, "dimensions": dimensions}).Printf("Enabled PQ. Waiting for shard ready.\n")
	case CompressionTypeSQ:
		log.Printf("Enabled SQ. Waiting for shard ready.\n")
	}

	start := time.Now()

	for {
		time.Sleep(3 * time.Second)
		diff := time.Since(start)
		if diff.Minutes() > 50 {
			log.Fatalf("Shard still not ready after 50 minutes, exiting..\n")
		}
		shards, err := client.Schema().ShardsGetter().WithClassName(cfg.ClassName).Do(context.Background())
		if err != nil || len(shards) == 0 {
			if weaviateErr, ok := err.(*fault.WeaviateClientError); ok {
				log.Warnf("Error getting schema: %v", weaviateErr.DerivedFromError)
			} else {
				log.Warnf("Error getting schema: %v", err)
			}
			continue
		}
		ready := true
		for _, shard := range shards {
			if shard.Status != "READY" {
				ready = false
			}
		}
		if ready {
			break
		}
	}

	endTime := time.Now()
	switch compressionType {
	case CompressionTypePQ:
		log.WithFields(log.Fields{"segments": segments, "dimensions": dimensions}).Printf("PQ Completed in %v\n", endTime.Sub(start))
	case CompressionTypeSQ:
		log.Printf("SQ Completed in %v\n", endTime.Sub(start))
	case CompressionTypeRQ:
		log.Printf("RQ Completed in %v\n", endTime.Sub(start))
	}
}

func convert1DChunk[D float32 | float64](input []D, dimensions int, batchRows int) [][]float32 {
	chunkData := make([][]float32, batchRows)
	for i := range chunkData {
		chunkData[i] = make([]float32, dimensions)
		for j := 0; j < dimensions; j++ {
			chunkData[i][j] = float32(input[i*dimensions+j])
		}
	}
	return chunkData
}

func getHDF5ByteSize(dataset *hdf5.Dataset) uint {
	datatype, err := dataset.Datatype()
	if err != nil {
		log.Fatalf("Unabled to read datatype\n")
	}

	// log.WithFields(log.Fields{"size": datatype.Size()}).Printf("Parsing HDF5 byte format\n")
	byteSize := datatype.Size()
	if byteSize != 4 && byteSize != 8 && byteSize != 16 {
		log.Fatalf("Unable to load dataset with byte size %d\n", byteSize)
	}
	return byteSize
}

// Load a large dataset from an hdf5 file and stream it to Weaviate
// startOffset and maxRecords are ignored if equal to 0
func loadHdf5Streaming(dataset *hdf5.Dataset, chunks chan<- Batch, cfg *Config, startOffset uint, maxRecords uint, filters []int) {
	dataspace := dataset.Space()
	dims, _, _ := dataspace.SimpleExtentDims()

	if len(dims) != 2 {
		log.Fatal("expected 2 dimensions")
	}

	byteSize := getHDF5ByteSize(dataset)

	rows := dims[0]
	dimensions := dims[1]

	// Handle offsetting the data for product quantization
	i := uint(0)
	if maxRecords != 0 && maxRecords < rows {
		rows = maxRecords
	}

	if startOffset != 0 && i < rows {
		i = startOffset
	}

	batchSize := uint(cfg.BatchSize)

	log.WithFields(log.Fields{"rows": rows, "dimensions": dimensions}).Printf(
		"Reading HDF5 dataset")

	memspace, err := hdf5.CreateSimpleDataspace([]uint{batchSize, dimensions}, []uint{batchSize, dimensions})
	if err != nil {
		log.Fatalf("Error creating memspace: %v", err)
	}
	defer memspace.Close()

	for ; i < rows; i += batchSize {

		batchRows := batchSize
		// handle final smaller batch
		if i+batchSize > rows {
			batchRows = rows - i
			memspace, err = hdf5.CreateSimpleDataspace([]uint{batchRows, dimensions}, []uint{batchRows, dimensions})
			if err != nil {
				log.Fatalf("Error creating final memspace: %v", err)
			}
		}

		offset := []uint{i, 0}
		count := []uint{batchRows, dimensions}

		if err := dataspace.SelectHyperslab(offset, nil, count, nil); err != nil {
			log.Fatalf("Error selecting hyperslab: %v", err)
		}

		var chunkData [][]float32

		if byteSize == 4 {
			chunkData1D := make([]float32, batchRows*dimensions)

			if err := dataset.ReadSubset(&chunkData1D, memspace, dataspace); err != nil {
				log.Printf("BatchRows = %d, i = %d, rows = %d", batchRows, i, rows)
				log.Fatalf("Error reading subset: %v", err)
			}

			chunkData = convert1DChunk[float32](chunkData1D, int(dimensions), int(batchRows))

		} else if byteSize == 8 {
			chunkData1D := make([]float64, batchRows*dimensions)

			if err := dataset.ReadSubset(&chunkData1D, memspace, dataspace); err != nil {
				log.Printf("BatchRows = %d, i = %d, rows = %d", batchRows, i, rows)
				log.Fatalf("Error reading subset: %v", err)
			}

			chunkData = convert1DChunk[float64](chunkData1D, int(dimensions), int(batchRows))

		}

		if (i+batchRows)%10000 == 0 {
			log.Printf("Imported %d/%d rows", i+batchRows, rows)
		}

		filter := []int{}
		if len(filters) > 0 {
			filter = filters[i : i+batchRows]
		}

		chunks <- Batch{Vectors: chunkData, Offset: int(i), Filters: filter}
	}
}

// Read an entire dataset from an hdf5 file at once
func loadHdf5Float32(file *hdf5.File, name string, cfg *Config) [][]float32 {
	dataset, err := file.OpenDataset(name)
	if err != nil {
		log.Fatalf("Error opening loadHdf5Float32 dataset: %v", err)
	}
	defer dataset.Close()
	dataspace := dataset.Space()
	dims, _, _ := dataspace.SimpleExtentDims()

	byteSize := getHDF5ByteSize(dataset)

	var rows uint
	var dimensions uint
	if cfg.MultiVectorDimensions != 0 {
		rows = dims[0]
		dimensions = uint(cfg.MultiVectorDimensions)
	} else {
		if len(dims) != 2 {
			log.Fatal("expected 2 dimensions")
		}
		rows = dims[0]
		dimensions = dims[1]
	}

	var chunkData [][]float32

	if byteSize == 4 {
		chunkData1D := make([]float32, rows*dimensions)
		dataset.Read(&chunkData1D)
		chunkData = convert1DChunk[float32](chunkData1D, int(dimensions), int(rows))
	} else if byteSize == 8 {
		chunkData1D := make([]float64, rows*dimensions)
		dataset.Read(&chunkData1D)
		chunkData = convert1DChunk[float64](chunkData1D, int(dimensions), int(rows))
	}

	return chunkData
}

func loadHdf5Categories(file *hdf5.File, name string) []int {
	dataset, err := file.OpenDataset(name)
	if err != nil {
		log.Fatalf("Error opening neighbors dataset: %v", err)
	}
	defer dataset.Close()

	dataspace := dataset.Space()
	dims, _, _ := dataspace.SimpleExtentDims()
	if len(dims) != 1 {
		log.Fatal("expected 1 dimension")
	}

	elements := dims[0]
	byteSize := getHDF5ByteSize(dataset)

	chunkData := make([]int, elements)

	if byteSize == 4 {
		chunkData32 := make([]int32, elements)
		dataset.Read(&chunkData32)
		for i := range chunkData {
			chunkData[i] = int(chunkData32[i])
		}
	} else if byteSize == 8 {
		dataset.Read(&chunkData)
	}

	return chunkData
}

// Read an entire dataset from an hdf5 file at once (neighbours)
func loadHdf5Neighbors(file *hdf5.File, name string) [][]int {
	dataset, err := file.OpenDataset(name)
	if err != nil {
		log.Fatalf("Error opening neighbors dataset: %v", err)
	}
	defer dataset.Close()
	dataspace := dataset.Space()
	dims, _, _ := dataspace.SimpleExtentDims()

	if len(dims) != 2 {
		log.Fatal("expected 2 dimensions")
	}

	rows := dims[0]
	dimensions := dims[1]

	byteSize := getHDF5ByteSize(dataset)

	chunkData := make([][]int, rows)

	if byteSize == 4 {
		chunkData1D := make([]int32, rows*dimensions)
		dataset.Read(&chunkData1D)
		for i := range chunkData {
			chunkData[i] = make([]int, dimensions)
			for j := uint(0); j < dimensions; j++ {
				chunkData[i][j] = int(chunkData1D[uint(i)*dimensions+j])
			}
		}
	} else if byteSize == 8 {
		chunkData1D := make([]int, rows*dimensions)
		dataset.Read(&chunkData1D)
		for i := range chunkData {
			chunkData[i] = chunkData1D[i*int(dimensions) : (i+1)*int(dimensions)]
		}
	}

	return chunkData
}

func calculateHdf5TrainExtent(file *hdf5.File, cfg *Config) (uint, uint) {
	dataset, err := file.OpenDataset("train")
	if err != nil {
		log.Fatalf("Error opening dataset: %v", err)
	}
	defer dataset.Close()
	dataspace := dataset.Space()
	extent, _, _ := dataspace.SimpleExtentDims()
	dimensions := extent[1]
	rows := extent[0]
	return rows, dimensions
}

func loadHdf5Train(file *hdf5.File, cfg *Config, offset uint, maxRows uint, updatePercent float32) uint {
	dataset, err := file.OpenDataset("train")
	if err != nil {
		log.Fatalf("Error opening dataset: %v", err)
	}
	defer dataset.Close()
	dataspace := dataset.Space()
	extent, _, _ := dataspace.SimpleExtentDims()
	var dimensions uint

	if cfg.MultiVectorDimensions == 0 {
		dimensions = extent[1]
	} else {
		dimensions = uint(cfg.MultiVectorDimensions)
	}

	filters := []int{}
	if cfg.Filter {
		filters = loadHdf5Categories(file, "train_categories")
	}

	chunks := make(chan Batch, 10)

	go func() {
		if cfg.MultiVectorDimensions > 0 {
			loadHdf5StreamingColbert(dataset, chunks, cfg, offset, maxRows, filters)
		} else {
			loadHdf5Streaming(dataset, chunks, cfg, offset, maxRows, filters)
		}
		close(chunks)
	}()

	var wg sync.WaitGroup

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Import workers will primary use the direct gRPC client
			// If triggering deletes before import, we need to use the normal go client
			grpcCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			httpOption := grpc.WithInsecure()
			if cfg.HttpScheme == "https" {
				creds := credentials.NewTLS(&tls.Config{
					InsecureSkipVerify: true,
				})
				httpOption = grpc.WithTransportCredentials(creds)
			}
			defer cancel()
			opts := []retry.CallOption{
				retry.WithBackoff(retry.BackoffExponential(100 * time.Millisecond)),
			}
			grpcConn, err := grpc.DialContext(grpcCtx, cfg.Origin, httpOption, grpc.WithUnaryInterceptor(retry.UnaryClientInterceptor(opts...)))
			if err != nil {
				log.Fatalf("Did not connect: %v", err)
			}
			defer grpcConn.Close()
			grpcClient := weaviategrpc.NewWeaviateClient(grpcConn)
			weaviateClient := createClient(cfg)

			for chunk := range chunks {
				if updatePercent > 0 {
					if rand.Float32() < updatePercent {
						deleteChunk(&chunk, weaviateClient, cfg)
						writeChunk(&chunk, &grpcClient, cfg)
					}
				} else {
					writeChunk(&chunk, &grpcClient, cfg)
				}
			}
		}()
	}

	wg.Wait()
	return dimensions
}

// Load an hdf5 file in the format of ann-benchmarks.com
// returns total time duration for load
func loadANNBenchmarksFile(file *hdf5.File, cfg *Config, client *weaviate.Client, maxRows uint) time.Duration {
	addTenantIfNeeded(cfg, client)
	startTime := time.Now()

	if cfg.PQ == "enabled" {
		dimensions := loadHdf5Train(file, cfg, 0, uint(cfg.TrainingLimit), 0)
		log.Printf("Pausing to enable PQ.")
		enableCompression(cfg, client, dimensions, CompressionTypePQ)
		loadHdf5Train(file, cfg, uint(cfg.TrainingLimit), 0, 0)

	} else if cfg.SQ == "enabled" {
		dimensions := loadHdf5Train(file, cfg, 0, uint(cfg.TrainingLimit), 0)
		log.Printf("Pausing to enable SQ.")
		enableCompression(cfg, client, dimensions, CompressionTypeSQ)
		loadHdf5Train(file, cfg, uint(cfg.TrainingLimit), 0, 0)

	} else if cfg.RQ == "enabled" {
		dimensions := loadHdf5Train(file, cfg, 0, uint(cfg.TrainingLimit), 0)
		log.Printf("Pausing to enable RQ.")
		enableCompression(cfg, client, dimensions, CompressionTypeRQ)
		loadHdf5Train(file, cfg, uint(cfg.TrainingLimit), 0, 0)
	} else {
		loadHdf5Train(file, cfg, 0, maxRows, 0)
	}
	endTime := time.Now()
	log.WithFields(log.Fields{"duration": endTime.Sub(startTime)}).Printf("Total load time\n")
	if !cfg.SkipAsyncReady {
		endTime = waitReady(cfg, client, startTime, 4*time.Hour, 1000)
	}
	return endTime.Sub(startTime)
}

// Load a dataset multiple time with different tenants
func loadHdf5MultiTenant(file *hdf5.File, cfg *Config, client *weaviate.Client) time.Duration {
	startTime := time.Now()

	for i := 0; i < cfg.NumTenants; i++ {
		cfg.Tenant = fmt.Sprintf("%d", i)
		loadANNBenchmarksFile(file, cfg, client, 0)
	}

	endTime := time.Now()
	log.WithFields(log.Fields{"duration": endTime.Sub(startTime)}).Printf("Multi-tenant load time\n")
	return endTime.Sub(startTime)
}

func parseEfValues(s string) ([]int, error) {
	strs := strings.Split(s, ",")
	nums := make([]int, len(strs))
	for i, str := range strs {
		num, err := strconv.Atoi(str)
		if err != nil {
			return nil, fmt.Errorf("error converting efArray '%s' to integer: %v", str, err)
		}
		nums[i] = num
	}
	return nums, nil
}

func runQueries(cfg *Config, importTime time.Duration, testData [][]float32, neighbors [][]int, filters []int) {
	runID := strconv.FormatInt(time.Now().Unix(), 10)

	// Generate filename with bench_ prefix and optional experiment name
	var filename string
	if cfg.ExperimentName != "" {
		filename = fmt.Sprintf("bench_%s_%s.json", cfg.ExperimentName, runID)
	} else {
		filename = fmt.Sprintf("bench_%s.json", runID)
	}

	efCandidates, err := parseEfValues(cfg.EfArray)
	if err != nil {
		log.Fatalf("Error parsing efArray, expected commas separated format \"16,32,64\" but:%v\n", err)
	}

	// Read once at this point (after import and compaction delay) to get accurate memory stats
	memstats := &Memstats{}
	if !cfg.SkipMemoryStats {
		memstats, err = readMemoryMetrics(cfg)
		if err != nil {
			log.Warnf("Error reading memory stats: %v", err)
			memstats = &Memstats{}
		}
	}

	client := createClient(cfg)

	var benchmarkResultsMap []map[string]interface{}
	for _, ef := range efCandidates {
		updateEf(ef, cfg, client)

		var result Results

		if cfg.QueryDuration > 0 {
			result = benchmarkANNDuration(*cfg, testData, neighbors, filters)
		} else {
			result = benchmarkANN(*cfg, testData, neighbors, filters)
		}

		log.WithFields(log.Fields{
			"mean": result.Mean, "qps": result.QueriesPerSecond, "recall": result.Recall, "ndcg": result.NDCG,
			"parallel": cfg.Parallel, "limit": cfg.Limit,
			"api": cfg.API, "ef": ef, "count": result.Total, "failed": result.Failed,
		}).Info("Benchmark result")

		dataset := filepath.Base(cfg.BenchmarkFile)

		var resultMap map[string]interface{}

		benchResult := ResultsJSONBenchmark{
			Api:              cfg.API,
			Ef:               ef,
			EfConstruction:   cfg.EfConstruction,
			MaxConnections:   cfg.MaxConnections,
			Mean:             result.Mean.Seconds(),
			P99Latency:       result.Percentiles[len(result.Percentiles)-1].Seconds(),
			QueriesPerSecond: result.QueriesPerSecond,
			Shards:           cfg.Shards,
			Parallelization:  cfg.Parallel,
			Limit:            cfg.Limit,
			ImportTime:       importTime.Seconds(),
			RunID:            runID,
			Dataset:          dataset,
			NDCG:             result.NDCG,
			Recall:           result.Recall,
			HeapAllocBytes:   memstats.HeapAllocBytes,
			HeapInuseBytes:   memstats.HeapInuseBytes,
			HeapSysBytes:     memstats.HeapSysBytes,
			Timestamp:        time.Now().Format(time.RFC3339),
		}

		jsonData, err := json.Marshal(benchResult)
		if err != nil {
			log.Fatalf("Error converting result to json")
		}

		if err := json.Unmarshal(jsonData, &resultMap); err != nil {
			log.Fatalf("Error converting json to map")
		}

		if cfg.LabelMap != nil {
			for key, value := range cfg.LabelMap {
				resultMap[key] = value
			}
		}

		benchmarkResultsMap = append(benchmarkResultsMap, resultMap)

	}

	// Run compression recall analysis if requested
	if cfg.CompressionRecallAnalysis {
		log.Info("Starting compression recall analysis")
		recallResults, err := measureCompressionRecall(cfg, client, testData, neighbors, filters)
		if err != nil {
			log.WithError(err).Error("Failed to measure compression recall")
		} else {
			log.WithField("results_count", len(recallResults)).Info("Compression recall analysis completed")

			// Save recall analysis results to file
			recallFile := fmt.Sprintf("compression_recall_%s_%s.json",
				cfg.ExperimentName, runID)
			if cfg.ExperimentName == "" {
				recallFile = fmt.Sprintf("compression_recall_%s.json", runID)
			}

			recallData, _ := json.MarshalIndent(recallResults, "", "  ")
			recallPath := fmt.Sprintf("./results/%s", recallFile)
			if err := os.WriteFile(recallPath, recallData, 0644); err != nil {
				log.WithError(err).Error("Failed to save compression recall analysis results")
			} else {
				log.WithField("file", recallPath).Info("Compression recall analysis results saved")
			}

			// Print summary
			log.Info("Compression Recall Analysis Summary:")
			for _, result := range recallResults {
				log.WithFields(log.Fields{
					"compression_type":   result.CompressionType,
					"recall":             result.Recall,
					"ndcg":               result.NDCG,
					"queries_per_second": result.QueriesPerSecond,
					"mean_latency_ms":    result.MeanLatency,
				}).Info("Recall measurement")
			}
		}
	}

	data, err := json.MarshalIndent(benchmarkResultsMap, "", "    ")
	if err != nil {
		log.Fatalf("Error marshaling benchmark results: %v", err)
	}

	os.Mkdir("./results", 0o755)

	// Use OutputFile if specified, otherwise use the generated filename in results directory
	var outputPath string
	if cfg.OutputFile != "" {
		outputPath = cfg.OutputFile
		// Ensure parent directory exists
		if dir := filepath.Dir(outputPath); dir != "." {
			os.MkdirAll(dir, 0o755)
		}
	} else {
		outputPath = fmt.Sprintf("./results/%s", filename)
	}

	err = os.WriteFile(outputPath, data, 0o644)
	if err != nil {
		log.Fatalf("Error writing benchmark results to file: %v", err)
	}

	log.WithFields(log.Fields{
		"output_path": outputPath,
		"filename":    filename,
		"experiment":  cfg.ExperimentName,
		"run_id":      runID,
	}).Info("Benchmark results saved to file")
	// Extract metrics from benchmark results
	metrics := make(map[string]float64)
	if len(benchmarkResultsMap) > 0 {
		// Take metrics from the first result as representative
		firstResult := benchmarkResultsMap[0]
		if qps, ok := firstResult["qps"].(float64); ok {
			metrics["qps"] = qps
		}
		if recall, ok := firstResult["recall"].(float64); ok {
			metrics["recall"] = recall
		}
		if latency, ok := firstResult["meanLatency"].(float64); ok {
			metrics["mean_latency"] = latency
		}
	}
}

var annBenchmarkCommand = &cobra.Command{
	Use:   "ann-benchmark",
	Short: "Benchmark ANN Benchmark style datasets",
	Long:  `Run a gRPC benchmark on an hdf5 file in the format of ann-benchmarks.com`,
	Run: func(cmd *cobra.Command, args []string) {
		cfg := globalConfig
		cfg.Mode = "ann-benchmark"

		if err := cfg.Validate(); err != nil {
			fatal(err)
		}

		cfg.parseLabels()

		memoryMonitor := NewMemoryMonitor(&cfg)
		memoryMonitor.Start()
		defer memoryMonitor.Stop()

		// Detect dataset format (currently only supports HDF5)
		format := "hdf5"
		var err error

		// Open HDF5 file for dataset operations
		var file *hdf5.File
		if !cfg.QueryOnly || !cfg.SkipQuery {
			file, err = hdf5.OpenFile(cfg.BenchmarkFile, hdf5.F_ACC_RDONLY)
			if err != nil {
				log.Fatalf("Error opening HDF5 file: %v", err)
			}
			defer file.Close()
		}

		log.WithFields(log.Fields{
			"file":   cfg.BenchmarkFile,
			"format": format,
		}).Info("Detected dataset format")

		// Run compression size analysis if requested (before creating client)
		if cfg.CompressionSizeAnalysis {
			log.Info("Starting compression size analysis")
			compressionResults, err := measureCompressionIndexSize(&cfg, nil)
			if err != nil {
				log.WithError(err).Error("Failed to measure compression index sizes")
			} else {
				log.WithField("results_count", len(compressionResults)).Info("Compression size analysis completed")

				// Save compression results to file
				compressionFile := fmt.Sprintf("compression_analysis_%s_%d.json",
					cfg.ExperimentName, time.Now().Unix())
				if cfg.ExperimentName == "" {
					compressionFile = fmt.Sprintf("compression_analysis_%d.json", time.Now().Unix())
				}

				// Ensure results directory exists
				os.Mkdir("./results", 0o755)

				compressionPath := fmt.Sprintf("./results/%s", compressionFile)
				compressionData, _ := json.MarshalIndent(compressionResults, "", "  ")
				if err := os.WriteFile(compressionPath, compressionData, 0644); err != nil {
					log.WithError(err).Error("Failed to save compression analysis results")
				} else {
					log.WithField("file", compressionPath).Info("Compression analysis results saved")
				}

				// Print summary
				log.Info("Compression Analysis Summary:")
				for _, result := range compressionResults {
					log.WithFields(log.Fields{
						"type":              result.CompressionType,
						"compression_ratio": fmt.Sprintf("%.2fx", result.CompressionRatio),
						"memory_mb":         fmt.Sprintf("%.1f MB", result.MemoryUsageMB),
						"size_reduction":    fmt.Sprintf("%.1f%%", (1-1/result.CompressionRatio)*100),
					}).Info("Compression result")
				}
			}

			// Exit early if only doing compression analysis
			if cfg.QueryOnly && cfg.SkipQuery {
				log.Info("Compression analysis completed, exiting")
				return
			}
		}

		client := createClient(&cfg)

		importTime := 0 * time.Second

		if !cfg.QueryOnly {

			if !cfg.ExistingSchema {
				createSchema(&cfg, client)
			}

			log.WithFields(log.Fields{
				"index": cfg.IndexType, "efC": cfg.EfConstruction, "m": cfg.MaxConnections, "shards": cfg.Shards,
				"distance": cfg.DistanceMetric, "dataset": cfg.BenchmarkFile,
			}).Info("Starting import")

			if cfg.NumTenants > 0 {
				importTime = loadHdf5MultiTenant(file, &cfg, client)
			} else {
				importTime = loadANNBenchmarksFile(file, &cfg, client, 0)
			}

			sleepDuration := time.Duration(cfg.QueryDelaySeconds) * time.Second
			log.Printf("Waiting for %s to allow for compaction etc\n", sleepDuration)
			time.Sleep(sleepDuration)
		}

		log.WithFields(log.Fields{
			"index": cfg.IndexType, "efC": cfg.EfConstruction, "m": cfg.MaxConnections, "shards": cfg.Shards,
			"distance": cfg.DistanceMetric, "dataset": cfg.BenchmarkFile,
		}).Info("Benchmark configuration")

		if cfg.SkipQuery {
			return
		}

		neighbors := loadHdf5Neighbors(file, "neighbors")
		var testData [][]float32
		if cfg.MultiVectorDimensions > 0 {
			testData = loadHdf5Colbert(file, "test", cfg.MultiVectorDimensions)
		} else {
			testData = loadHdf5Float32(file, "test", &cfg)
		}

		testFilters := make([]int, 0)
		if cfg.Filter {
			testFilters = loadHdf5Categories(file, "test_categories")
		}

		runQueries(&cfg, importTime, testData, neighbors, testFilters)

		if cfg.performUpdates() {

			totalRowCount, _ := calculateHdf5TrainExtent(file, &cfg)
			updateRowCount := uint(math.Floor(float64(totalRowCount) * cfg.UpdatePercentage))

			log.Printf("Performing %d update iterations\n", cfg.UpdateIterations)

			for i := 0; i < cfg.UpdateIterations; i++ {

				startTime := time.Now()

				if cfg.UpdateRandomized {
					loadHdf5Train(file, &cfg, 0, 0, float32(cfg.UpdatePercentage))
				} else {
					deleteUuidRange(&cfg, client, 0, int(updateRowCount))
					loadHdf5Train(file, &cfg, 0, updateRowCount, 0)
				}

				log.WithFields(log.Fields{"duration": time.Since(startTime)}).Printf("Total delete and update time\n")

				if !cfg.SkipTombstonesEmpty {
					err := waitTombstonesEmpty(&cfg)
					if err != nil {
						log.Fatalf("Error waiting for tombstones to be empty: %v", err)
					}
				}
				if !cfg.SkipAsyncReady {
					startTime := time.Now()
					waitReady(&cfg, client, startTime, 30*time.Minute, 1000)
				}

				runQueries(&cfg, importTime, testData, neighbors, testFilters)

			}

		}
	},
}

func initAnnBenchmark() {
	rootCmd.AddCommand(annBenchmarkCommand)

	numCPU := runtime.NumCPU()

	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.Labels,
		"labels", "", "Labels of format key1=value1,key2=value2,...")
	annBenchmarkCommand.PersistentFlags().StringVarP(&globalConfig.BenchmarkFile,
		"vectors", "v", "", "Path to the hdf5 file containing the vectors")
	annBenchmarkCommand.PersistentFlags().StringVarP(&globalConfig.ClassName,
		"className", "c", "Vector", "Class name for testing")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.NamedVector,
		"namedVector", "", "Named vector")
	annBenchmarkCommand.PersistentFlags().StringVarP(&globalConfig.DistanceMetric,
		"distance", "d", "", "Set distance metric (mandatory)")
	annBenchmarkCommand.PersistentFlags().BoolVarP(&globalConfig.QueryOnly,
		"query", "q", false, "Do not import data and only run query tests")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.QueryDuration,
		"queryDuration", 0, "Instead of querying the test dataset once, query for the specified duration in seconds (default 0)")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.BQ,
		"bq", false, "Set BQ")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.Cache,
		"cache", false, "Set cache")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.RescoreLimit,
		"rescoreLimit", -1, "Rescore limit. If not set, it will be set by Weaviate automatically when rescoring is enabled")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.PQ,
		"pq", "disabled", "Set PQ (disabled, auto, or enabled) (default disabled)")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.SQ,
		"sq", "disabled", "Set SQ (disabled, auto, or enabled) (default disabled)")
	annBenchmarkCommand.PersistentFlags().UintVar(&globalConfig.PQRatio,
		"pqRatio", 4, "Set PQ segments = dimensions / ratio (must divide evenly default 4)")
	annBenchmarkCommand.PersistentFlags().UintVar(&globalConfig.PQSegments,
		"pqSegments", 256, "Set PQ segments")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.RQ,
		"rq", "disabled", "Set RQ (disabled, auto, or enabled) (default disabled)")
	annBenchmarkCommand.PersistentFlags().UintVar(&globalConfig.RQBits,
		"rqBits", 8, "Set RQ bits (default 8)")
	annBenchmarkCommand.PersistentFlags().IntVarP(&globalConfig.MultiVectorDimensions,
		"multiVector", "m", 0, "Enable multi-dimensional vectors with the specified number of dimensions")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.MuveraEnabled,
		"muveraEnabled", false, "Enable muvera")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.MuveraKSim,
		"muveraKSim", 4, "Set muvera ksim parameter")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.MuveraDProjections,
		"muveraDProjections", 16, "Set muvera dprojections parameter")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.MuveraRepetition,
		"muveraRepetition", 20, "Set muvera repetition parameter")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.SkipQuery,
		"skipQuery", false, "Only import data and skip query tests")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.SkipAsyncReady,
		"skipAsyncReady", false, "Skip async ready (default false)")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.SkipMemoryStats,
		"skipMemoryStats", false, "Skip memory stats (default false)")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.SkipTombstonesEmpty,
		"skipTombstonesEmpty", false, "Skip waiting for tombstone to be empty after update (default false)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.TrainingLimit,
		"trainingLimit", 100000, "Set PQ trainingLimit (default 100000)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.EfConstruction,
		"efConstruction", 256, "Set Weaviate efConstruction parameter (default 256)")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.EfArray,
		"efArray", "16,24,32,48,64,96,128,256,512", "Array of ef parameters as comma separated list")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.IndexType,
		"indexType", "hnsw", "Index type (hnsw or flat)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.MaxConnections,
		"maxConnections", 16, "Set Weaviate efConstruction parameter (default 16)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.Shards,
		"shards", 1, "Set number of Weaviate shards")
	annBenchmarkCommand.PersistentFlags().IntVarP(&globalConfig.BatchSize,
		"batchSize", "b", 1000, "Batch size for insert operations")
	annBenchmarkCommand.PersistentFlags().IntVarP(&globalConfig.Parallel,
		"parallel", "p", numCPU, "Set the number of parallel threads which send queries")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.ExistingSchema,
		"existingSchema", false, "Leave the schema as-is (default false)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.NumTenants,
		"numTenants", 0, "Number of tenants to use (default 0)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.StartTenantNum,
		"startTenant", 0, "Tenant # to start at if using multiple tenants (default 0)")
	annBenchmarkCommand.PersistentFlags().StringVarP(&globalConfig.API,
		"api", "a", "grpc", "The API to use on benchmarks")
	annBenchmarkCommand.PersistentFlags().StringVarP(&globalConfig.Origin,
		"grpcOrigin", "u", "localhost:50051", "The gRPC origin that Weaviate is running at")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.HttpOrigin,
		"httpOrigin", "localhost:8080", "The http origin for Weaviate (only used if grpc enabled)")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.HttpScheme,
		"httpScheme", "http", "The http scheme (http or https)")
	annBenchmarkCommand.PersistentFlags().StringVarP(&globalConfig.OutputFormat,
		"format", "f", "text", "Output format, one of [text, json]")
	annBenchmarkCommand.PersistentFlags().IntVarP(&globalConfig.Limit,
		"limit", "l", 10, "Set the query limit / k (default 10)")
	annBenchmarkCommand.PersistentFlags().Float64Var(&globalConfig.UpdatePercentage,
		"updatePercentage", 0.0, "After loading the dataset, update the specified percentage of vectors")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.UpdateRandomized,
		"updateRandomized", false, "Whether to randomize which vectors are updated (default false)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.UpdateIterations,
		"updateIterations", 1, "Number of iterations to update the dataset if updatePercentage is set")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.CleanupIntervalSeconds,
		"cleanupIntervalSeconds", 300, "HNSW cleanup interval seconds (default 300)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.QueryDelaySeconds,
		"queryDelaySeconds", 30, "How long to wait before querying (default 30)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.Offset,
		"offset", 0, "Offset for uuids (useful to load the same dataset multiple times)")
	annBenchmarkCommand.PersistentFlags().StringVarP(&globalConfig.OutputFile,
		"output", "o", "", "Filename for an output file. If none provided, output to stdout only")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.DynamicThreshold,
		"dynamicThreshold", 10_000, "Threshold to trigger the update in the dynamic index (default 10 000)")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.Filter,
		"filter", false, "Whether to use filtering for the dataset (default false)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.FlatSearchCutoff,
		"flatSearchCutoff", 40000, "Flat search cut off (default 40 000)")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.FilterStrategy,
		"filterStrategy", "sweeping", "Use a different filter strategy (options are sweeping or acorn)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.ReplicationFactor,
		"replicationFactor", 1, "Replication factor (default 1)")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.AsyncReplicationEnabled,
		"asyncReplicationEnabled", false, "Enable asynchronous replication (default false)")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.MemoryMonitoringEnabled,
		"memoryMonitoringEnabled", false, "Enable continuous memory monitoring (default false)")
	annBenchmarkCommand.PersistentFlags().IntVar(&globalConfig.MemoryMonitoringInterval,
		"memoryMonitoringInterval", 5, "Memory monitoring interval in seconds (default 5)")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.MemoryMonitoringFile,
		"memoryMonitoringFile", "", "Memory monitoring output file name (default: memory_metrics_<timestamp>.json)")
	annBenchmarkCommand.PersistentFlags().StringVar(&globalConfig.ExperimentName,
		"experimentName", "", "Name for this experiment (used in output file naming and logging)")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.CompressionSizeAnalysis,
		"compressionSizeAnalysis", false, "Enable compression index size measurement analysis")
	annBenchmarkCommand.PersistentFlags().BoolVar(&globalConfig.CompressionRecallAnalysis,
		"compressionRecallAnalysis", false, "Enable compression recall measurement and comparison analysis")
}

func benchmarkANN(cfg Config, queries Queries, neighbors Neighbors, filters []int) Results {
	cfg.Queries = len(queries)

	i := 0
	return benchmark(cfg, func(className string) QueryWithNeighbors {
		defer func() { i++ }()

		tenant := ""
		if cfg.NumTenants > 0 {
			tenant = fmt.Sprint(rand.Intn(cfg.NumTenants))
		}
		filter := -1
		if cfg.Filter {
			filter = filters[i]
		}

		return QueryWithNeighbors{
			Query:     nearVectorQueryGrpc(&cfg, queries[i], tenant, filter),
			Neighbors: neighbors[i],
		}
	})
}

type Number interface {
	constraints.Float | constraints.Integer
}

func median[T Number](data []T) float64 {
	dataCopy := make([]T, len(data))
	copy(dataCopy, data)

	slices.Sort(dataCopy)

	var median float64
	l := len(dataCopy)
	if l == 0 {
		return 0
	} else if l%2 == 0 {
		median = float64((dataCopy[l/2-1] + dataCopy[l/2]) / 2.0)
	} else {
		median = float64(dataCopy[l/2])
	}

	return median
}

type sampledResults struct {
	Min              []time.Duration
	Max              []time.Duration
	Mean             []time.Duration
	Took             []time.Duration
	QueriesPerSecond []float64
	Recall           []float64
	NDCG             []float64
	Results          []Results
}

func benchmarkANNDuration(cfg Config, queries Queries, neighbors Neighbors, filters []int) Results {
	cfg.Queries = len(queries)

	var samples sampledResults

	startTime := time.Now()

	var results Results

	for time.Since(startTime) < time.Duration(cfg.QueryDuration)*time.Second {
		results = benchmarkANN(cfg, queries, neighbors, filters)
		samples.Min = append(samples.Min, results.Min)
		samples.Max = append(samples.Max, results.Max)
		samples.Mean = append(samples.Mean, results.Mean)
		samples.Took = append(samples.Took, results.Took)
		samples.QueriesPerSecond = append(samples.QueriesPerSecond, results.QueriesPerSecond)
		samples.NDCG = append(samples.NDCG, results.NDCG)
		samples.Recall = append(samples.Recall, results.Recall)
		samples.Results = append(samples.Results, results)
	}

	var medianResult Results

	medianResult.Min = time.Duration(median(samples.Min))
	medianResult.Max = time.Duration(median(samples.Max))
	medianResult.Mean = time.Duration(median(samples.Mean))
	medianResult.Took = time.Duration(median(samples.Took))
	medianResult.QueriesPerSecond = median(samples.QueriesPerSecond)
	medianResult.Percentiles = results.Percentiles
	medianResult.PercentilesLabels = results.PercentilesLabels
	medianResult.Total = results.Total
	medianResult.Successful = results.Successful
	medianResult.Failed = results.Failed
	medianResult.Parallelization = cfg.Parallel
	medianResult.Recall = median(samples.Recall)
	medianResult.NDCG = median(samples.NDCG)

	return medianResult
}

// CompressionSizeInfo holds information about index sizes under different compression settings
type CompressionSizeInfo struct {
	CompressionType  string  `json:"compressionType"`
	UncompressedSize int64   `json:"uncompressedSizeBytes"`
	CompressedSize   int64   `json:"compressedSizeBytes"`
	CompressionRatio float64 `json:"compressionRatio"`
	VectorCount      int     `json:"vectorCount"`
	Dimensions       int     `json:"dimensions"`
	SegmentCount     int     `json:"segmentCount,omitempty"`
	TrainingLimit    int     `json:"trainingLimit,omitempty"`
	RQBits           int     `json:"rqBits,omitempty"`
	PQRatio          int     `json:"pqRatio,omitempty"`
	MemoryUsageMB    float64 `json:"memoryUsageMB"`
}

// CompressionRecallInfo holds recall measurement results for different compression configurations
type CompressionRecallInfo struct {
	CompressionType  string  `json:"compressionType"`
	Recall           float64 `json:"recall"`
	NDCG             float64 `json:"ndcg"`
	QueriesPerSecond float64 `json:"queriesPerSecond"`
	MeanLatency      float64 `json:"meanLatencyMs"`
	P99Latency       float64 `json:"p99LatencyMs"`
	VectorCount      int     `json:"vectorCount"`
	Dimensions       int     `json:"dimensions"`
	QueryCount       int     `json:"queryCount"`
	PQRatio          int     `json:"pqRatio,omitempty"`
	RQBits           int     `json:"rqBits,omitempty"`
	TrainingLimit    int     `json:"trainingLimit,omitempty"`
}

// measureCompressionIndexSize measures the index size for different compression configurations
func measureCompressionIndexSize(cfg *Config, client *weaviate.Client) ([]CompressionSizeInfo, error) {
	log.Info("Starting compression index size measurement")

	var compressionResults []CompressionSizeInfo

	// Get vector dimensions first by opening the HDF5 file
	file, err := hdf5.OpenFile(cfg.BenchmarkFile, hdf5.F_ACC_RDONLY)
	if err != nil {
		return nil, fmt.Errorf("failed to open HDF5 file: %v", err)
	}
	defer file.Close()

	// Load train vectors to get dimensions
	vectors := loadHdf5Float32(file, "train", cfg)
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors found in dataset")
	}

	vectorCount := len(vectors)
	dimensions := len(vectors[0])

	log.WithFields(log.Fields{
		"vector_count": vectorCount,
		"dimensions":   dimensions,
	}).Info("Dataset information")

	// Test configurations to benchmark
	compressionConfigs := []struct {
		name            string
		compressionType CompressionType
		enabled         bool
		pqRatio         int
		rqBits          int
	}{
		{"none", CompressionTypePQ, false, 0, 0},
		{"pq_ratio_2", CompressionTypePQ, true, 2, 0},
		{"pq_ratio_4", CompressionTypePQ, true, 4, 0},
		{"pq_ratio_8", CompressionTypePQ, true, 8, 0},
		{"rq_4bits", CompressionTypeRQ, true, 0, 4},
		{"rq_6bits", CompressionTypeRQ, true, 0, 6},
		{"rq_8bits", CompressionTypeRQ, true, 0, 8},
		{"sq", CompressionTypeSQ, true, 0, 0},
	}

	for _, config := range compressionConfigs {
		log.WithField("config", config.name).Info("Testing compression configuration")

		// Calculate uncompressed size (baseline)
		uncompressedSize := int64(vectorCount * dimensions * 4) // 4 bytes per float32

		sizeInfo := CompressionSizeInfo{
			CompressionType:  config.name,
			UncompressedSize: uncompressedSize,
			VectorCount:      vectorCount,
			Dimensions:       dimensions,
			TrainingLimit:    cfg.TrainingLimit,
		}

		if config.enabled {
			switch config.compressionType {
			case CompressionTypePQ:
				sizeInfo.PQRatio = config.pqRatio
				// PQ compressed size calculation: (dimensions / ratio) * vector_count
				if dimensions%config.pqRatio == 0 {
					sizeInfo.CompressedSize = int64((dimensions / config.pqRatio) * vectorCount)
					sizeInfo.SegmentCount = dimensions / config.pqRatio
				} else {
					log.WithField("config", config.name).Warn("Dimensions not divisible by PQ ratio, skipping")
					continue
				}
			case CompressionTypeRQ:
				sizeInfo.RQBits = config.rqBits
				// RQ compressed size: (bits / 8) * dimensions * vector_count
				sizeInfo.CompressedSize = int64((config.rqBits * dimensions * vectorCount) / 8)
			case CompressionTypeSQ:
				// SQ typically uses 1 byte per dimension
				sizeInfo.CompressedSize = int64(dimensions * vectorCount)
			}
		} else {
			sizeInfo.CompressedSize = uncompressedSize
		}

		// Calculate compression ratio
		if sizeInfo.UncompressedSize > 0 {
			sizeInfo.CompressionRatio = float64(sizeInfo.UncompressedSize) / float64(sizeInfo.CompressedSize)
		}

		// Estimate memory usage in MB
		sizeInfo.MemoryUsageMB = float64(sizeInfo.CompressedSize) / (1024 * 1024)

		log.WithFields(log.Fields{
			"compression_type":  sizeInfo.CompressionType,
			"uncompressed_size": sizeInfo.UncompressedSize,
			"compressed_size":   sizeInfo.CompressedSize,
			"compression_ratio": sizeInfo.CompressionRatio,
			"memory_usage_mb":   sizeInfo.MemoryUsageMB,
		}).Info("Compression measurement completed")

		compressionResults = append(compressionResults, sizeInfo)
	}

	return compressionResults, nil
}

// measureRealIndexSize creates an actual index and measures its real size
func measureRealIndexSize(cfg *Config, client *weaviate.Client, className string, vectors [][]float32, config struct {
	name            string
	compressionType CompressionType
	enabled         bool
	pqRatio         int
	rqBits          int
}) (int64, error) {

	log.WithField("class_name", className).Info("Creating test schema for real size measurement")

	// Create schema
	schema := &models.Class{
		Class: className,
		Properties: []*models.Property{
			{
				Name:     "content",
				DataType: []string{"text"},
			},
		},
		VectorIndexConfig: map[string]interface{}{
			"distance": cfg.DistanceMetric,
		},
	}

	// Apply compression settings
	if config.enabled {
		vectorIndexConfig := schema.VectorIndexConfig.(map[string]interface{})

		switch config.compressionType {
		case CompressionTypePQ:
			vectorIndexConfig["pq"] = map[string]interface{}{
				"enabled":       true,
				"trainingLimit": cfg.TrainingLimit,
				"segments":      cfg.Dimensions / config.pqRatio,
			}
		case CompressionTypeRQ:
			vectorIndexConfig["rq"] = map[string]interface{}{
				"enabled": true,
				"bits":    config.rqBits,
			}
		case CompressionTypeSQ:
			vectorIndexConfig["sq"] = map[string]interface{}{
				"enabled": true,
			}
		}

		schema.VectorIndexConfig = vectorIndexConfig
	}

	// Create the class
	if err := client.Schema().ClassCreator().WithClass(schema).Do(context.Background()); err != nil {
		return 0, fmt.Errorf("failed to create test class: %v", err)
	}

	// Import a subset of vectors for testing (limit to avoid long execution time)
	maxVectors := 1000
	if len(vectors) > maxVectors {
		vectors = vectors[:maxVectors]
	}

	// Import vectors
	batch := client.Batch().ObjectsBatcher()
	for i, vector := range vectors {
		object := &models.Object{
			Class: className,
			Properties: map[string]interface{}{
				"content": fmt.Sprintf("test_vector_%d", i),
			},
			Vector: vector,
		}
		batch = batch.WithObject(object)

		if (i+1)%100 == 0 || i == len(vectors)-1 {
			if _, err := batch.Do(context.Background()); err != nil {
				log.WithError(err).Warn("Failed to import batch")
			}
			batch = client.Batch().ObjectsBatcher()
		}
	}

	// Wait for indexing to complete
	time.Sleep(5 * time.Second)

	// Get index statistics (this is a simplified approach)
	// In a real implementation, you'd query Weaviate's metrics endpoint
	// For now, we'll estimate based on theoretical compression
	estimatedSize := int64(len(vectors) * len(vectors[0]) * 4) // baseline

	switch config.compressionType {
	case CompressionTypePQ:
		estimatedSize = int64((len(vectors[0]) / config.pqRatio) * len(vectors))
	case CompressionTypeRQ:
		estimatedSize = int64((config.rqBits * len(vectors[0]) * len(vectors)) / 8)
	case CompressionTypeSQ:
		estimatedSize = int64(len(vectors[0]) * len(vectors))
	}

	// Clean up the test class
	if err := client.Schema().ClassDeleter().WithClassName(className).Do(context.Background()); err != nil {
		log.WithError(err).Warn("Failed to delete test class")
	}

	return estimatedSize, nil
}

// measureCompressionRecall measures recall across different compression configurations
func measureCompressionRecall(cfg *Config, client *weaviate.Client, queries Queries, neighbors Neighbors, filters []int) ([]CompressionRecallInfo, error) {
	log.Info("Starting compression recall analysis")

	var recallResults []CompressionRecallInfo

	// Get vector dimensions from dataset
	file, err := hdf5.OpenFile(cfg.BenchmarkFile, hdf5.F_ACC_RDONLY)
	if err != nil {
		return nil, fmt.Errorf("failed to open HDF5 file: %v", err)
	}
	defer file.Close()

	vectors := loadHdf5Float32(file, "train", cfg)
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors found in dataset")
	}

	vectorCount := len(vectors)
	dimensions := len(vectors[0])
	queryCount := len(queries)

	log.WithFields(log.Fields{
		"vector_count": vectorCount,
		"dimensions":   dimensions,
		"query_count":  queryCount,
	}).Info("Dataset information for recall analysis")

	// Test configurations for recall measurement
	compressionConfigs := []struct {
		name            string
		compressionType CompressionType
		enabled         bool
		pqRatio         int
		rqBits          int
	}{
		{"none", CompressionTypePQ, false, 0, 0},
		{"pq_ratio_2", CompressionTypePQ, true, 2, 0},
		{"pq_ratio_4", CompressionTypePQ, true, 4, 0},
		{"pq_ratio_8", CompressionTypePQ, true, 8, 0},
		{"rq_4bits", CompressionTypeRQ, true, 0, 4},
		{"rq_6bits", CompressionTypeRQ, true, 0, 6},
		{"rq_8bits", CompressionTypeRQ, true, 0, 8},
		{"sq", CompressionTypeSQ, true, 0, 0},
	}

	for _, config := range compressionConfigs {
		log.WithField("config", config.name).Info("Testing compression configuration for recall")

		// Skip configurations that don't divide evenly for PQ
		if config.enabled && config.compressionType == CompressionTypePQ && dimensions%config.pqRatio != 0 {
			log.WithField("config", config.name).Warn("Dimensions not divisible by PQ ratio, skipping")
			continue
		}

		// Create a unique class name for this test
		className := fmt.Sprintf("RecallTest_%s_%d", strings.ReplaceAll(config.name, "_", ""), time.Now().Unix()%10000)

		// Create temporary config for this compression type
		tempCfg := *cfg
		tempCfg.ClassName = className

		// Set compression parameters in the temp config
		if config.enabled {
			switch config.compressionType {
			case CompressionTypePQ:
				tempCfg.PQ = "enabled"
				tempCfg.PQRatio = uint(config.pqRatio)
			case CompressionTypeRQ:
				tempCfg.RQ = "enabled"
				tempCfg.RQBits = uint(config.rqBits)
			case CompressionTypeSQ:
				tempCfg.SQ = "enabled"
			}
		} else {
			tempCfg.PQ = "disabled"
			tempCfg.RQ = "disabled"
			tempCfg.SQ = "disabled"
		}

		// Create schema for this configuration
		if err := createClass(&tempCfg, client); err != nil {
			log.WithError(err).WithField("config", config.name).Error("Failed to create class for recall test")
			continue
		}

		// Import vectors
		log.WithField("config", config.name).Info("Importing vectors for recall test")
		if err := importVectors(&tempCfg, vectors, client); err != nil {
			log.WithError(err).WithField("config", config.name).Error("Failed to import vectors")
			// Clean up before continuing
			cleanupClass(&tempCfg, client)
			continue
		}

		// Wait for indexing to complete
		log.WithField("config", config.name).Info("Waiting for indexing to complete")
		time.Sleep(10 * time.Second)

		// Run benchmark to measure recall
		log.WithField("config", config.name).Info("Running recall benchmark")
		results := benchmarkANN(tempCfg, queries, neighbors, filters)

		// Create recall info
		recallInfo := CompressionRecallInfo{
			CompressionType:  config.name,
			Recall:           results.Recall,
			NDCG:             results.NDCG,
			QueriesPerSecond: results.QueriesPerSecond,
			MeanLatency:      float64(results.Mean.Nanoseconds()) / 1e6, // Convert to milliseconds
			P99Latency:       float64(results.Percentiles[len(results.Percentiles)-1].Nanoseconds()) / 1e6,
			VectorCount:      vectorCount,
			Dimensions:       dimensions,
			QueryCount:       queryCount,
			TrainingLimit:    cfg.TrainingLimit,
		}

		// Add compression-specific parameters
		if config.enabled {
			switch config.compressionType {
			case CompressionTypePQ:
				recallInfo.PQRatio = config.pqRatio
			case CompressionTypeRQ:
				recallInfo.RQBits = config.rqBits
			}
		}

		log.WithFields(log.Fields{
			"compression_type":   recallInfo.CompressionType,
			"recall":             recallInfo.Recall,
			"ndcg":               recallInfo.NDCG,
			"queries_per_second": recallInfo.QueriesPerSecond,
			"mean_latency_ms":    recallInfo.MeanLatency,
		}).Info("Compression recall measurement completed")

		recallResults = append(recallResults, recallInfo)

		// Clean up the test class
		cleanupClass(&tempCfg, client)

		// Brief pause between configurations
		time.Sleep(2 * time.Second)
	}

	return recallResults, nil
}

// Helper function to clean up test class
func cleanupClass(cfg *Config, client *weaviate.Client) {
	if err := client.Schema().ClassDeleter().WithClassName(cfg.ClassName).Do(context.Background()); err != nil {
		log.WithError(err).WithField("class", cfg.ClassName).Warn("Failed to delete test class")
	}
}

// Helper function to create a class with the specified compression configuration
func createClass(cfg *Config, client *weaviate.Client) error {
	// Create schema
	schema := &models.Class{
		Class: cfg.ClassName,
		Properties: []*models.Property{
			{
				Name:     "content",
				DataType: []string{"text"},
			},
		},
		VectorIndexConfig: map[string]interface{}{
			"distance":       cfg.DistanceMetric,
			"efConstruction": cfg.EfConstruction,
			"maxConnections": cfg.MaxConnections,
		},
	}

	// Apply compression settings
	vectorIndexConfig := schema.VectorIndexConfig.(map[string]interface{})

	if cfg.PQ == "enabled" {
		vectorIndexConfig["pq"] = map[string]interface{}{
			"enabled":       true,
			"trainingLimit": cfg.TrainingLimit,
			"segments":      cfg.Dimensions / int(cfg.PQRatio),
		}
	}

	if cfg.RQ == "enabled" {
		vectorIndexConfig["rq"] = map[string]interface{}{
			"enabled": true,
			"bits":    cfg.RQBits,
		}
	}

	if cfg.SQ == "enabled" {
		vectorIndexConfig["sq"] = map[string]interface{}{
			"enabled": true,
		}
	}

	schema.VectorIndexConfig = vectorIndexConfig

	// Create the class
	return client.Schema().ClassCreator().WithClass(schema).Do(context.Background())
}

// Helper function to import vectors into a class
func importVectors(cfg *Config, vectors [][]float32, client *weaviate.Client) error {
	// Limit the number of vectors for testing to avoid long execution times
	maxVectors := 5000
	if len(vectors) > maxVectors {
		vectors = vectors[:maxVectors]
	}

	log.WithField("vector_count", len(vectors)).Info("Importing vectors for recall test")

	// Import vectors in batches
	batchSize := 100
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		batch := client.Batch().ObjectsBatcher()
		for j := i; j < end; j++ {
			object := &models.Object{
				Class: cfg.ClassName,
				ID:    strfmt.UUID(uuidFromInt(j)),
				Properties: map[string]interface{}{
					"content": fmt.Sprintf("test_vector_%d", j),
				},
				Vector: vectors[j],
			}
			batch = batch.WithObject(object)
		}

		if _, err := batch.Do(context.Background()); err != nil {
			return fmt.Errorf("failed to import batch %d-%d: %v", i, end, err)
		}
	}

	return nil
}
