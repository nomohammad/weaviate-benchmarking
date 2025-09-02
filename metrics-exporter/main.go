package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/spf13/cobra"
)

const (
	namespace = "benchmark"
)

type MetricData struct {
	API            string  `json:"api"`
	Branch         string  `json:"branch"`
	DatasetFile    string  `json:"dataset_file"`
	EF             int     `json:"ef"`
	EFConstruction int     `json:"efConstruction"`
	Limit          int     `json:"limit"`
	MaxConnections int     `json:"maxConnections"`
	MeanLatency    float64 `json:"meanLatency"`
	P99Latency     float64 `json:"p99Latency"`
	QPS            float64 `json:"qps"`
	Recall         float64 `json:"recall"`
	Shards         int     `json:"shards"`
	ImportTime     float64 `json:"importTime"`
	HeapAllocBytes float64 `json:"heap_alloc_bytes"`
	HeapInuseBytes float64 `json:"heap_inuse_bytes"`
	HeapSysBytes   float64 `json:"heap_sys_bytes"`
	TestID         string  `json:"run_id"`
}

// CompressionSizeInfo holds information about index sizes under different compression settings
type CompressionSizeInfo struct {
	CompressionType   string  `json:"compressionType"`
	UncompressedSize  int64   `json:"uncompressedSizeBytes"`
	CompressedSize    int64   `json:"compressedSizeBytes"`
	CompressionRatio  float64 `json:"compressionRatio"`
	VectorCount       int     `json:"vectorCount"`
	Dimensions        int     `json:"dimensions"`
	SegmentCount      int     `json:"segmentCount,omitempty"`
	TrainingLimit     int     `json:"trainingLimit,omitempty"`
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
}

type Exporter struct {
	metrics map[string]*prometheus.GaugeVec
}

func NewExporter() *Exporter {
	return &Exporter{
		metrics: make(map[string]*prometheus.GaugeVec),
	}
}

func (e *Exporter) initializeMetrics() {
	labels := []string{"branch", "dataset", "ef_construction", "max_connections", "limit", "ef", "shards", "run_id"}

	metricNames := []struct {
		name string
		help string
	}{
		{"latency_mean", "Mean latency of queries"},
		{"latency_p99", "99th percentile latency of queries"},
		{"qps", "Queries per second"},
		{"recall", "Recall metric"},
		{"heap_alloc_bytes", "Heap alloc bytes"},
		{"heap_sys_bytes", "Heap sys bytes"},
		{"heap_inuse_bytes", "Heap inuse bytes"},
		{"import_time", "Import time"},
		{"run_id", "Test ID"},
	}

	for _, metric := range metricNames {
		e.metrics[metric.name] = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      metric.name,
				Help:      metric.help,
			},
			labels,
		)
	}

	// Add compression-specific metrics with compression_type label
	compressionLabels := []string{"compression_type", "dataset"}

	compressionMetrics := []struct {
		name string
		help string
	}{
		{"compression_ratio", "Compression ratio achieved by different compression types"},
		{"index_size_bytes", "Index size in bytes for different compression types"},
		{"uncompressed_size_bytes", "Uncompressed index size in bytes"},
		{"memory_usage_mb", "Memory usage in MB for different compression types"},
		{"recall_by_compression", "Recall metric by compression type"},
		{"qps_by_compression", "Queries per second by compression type"},
		{"latency_by_compression", "Mean latency by compression type"},
	}

	for _, metric := range compressionMetrics {
		e.metrics[metric.name] = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      metric.name,
				Help:      metric.help,
			},
			compressionLabels,
		)
	}
}

func (e *Exporter) processJSONFile(filepath string) error {
	content, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("error reading file %s: %v", filepath, err)
	}

	// Determine file type by filename pattern first, then by content
	baseName := filepath
	if lastSlash := strings.LastIndex(filepath, "/"); lastSlash != -1 {
		baseName = filepath[lastSlash+1:]
	}

	// Try compression recall files first (check for recall field in filename or content)
	if strings.Contains(baseName, "recall") || strings.Contains(baseName, "performance") {
		if err := e.processCompressionRecallFile(content, filepath); err == nil {
			return nil
		}
	}

	// Try compression size files (check for "compression_analysis" pattern)
	if strings.Contains(baseName, "compression_analysis") || strings.Contains(baseName, "compression") {
		if err := e.processCompressionSizeFile(content, filepath); err == nil {
			return nil
		}
	}

	// Try compression recall files if size failed
	if err := e.processCompressionRecallFile(content, filepath); err == nil {
		return nil
	}

	// Try compression size files if recall failed
	if err := e.processCompressionSizeFile(content, filepath); err == nil {
		return nil
	}

	// Fall back to regular metrics processing
	return e.processRegularMetricsFile(content, filepath)
}

func (e *Exporter) processCompressionSizeFile(content []byte, filepath string) error {
	var compressionData []CompressionSizeInfo
	if err := json.Unmarshal(content, &compressionData); err != nil {
		return err
	}

	// Validate that this is actually compression size data
	if len(compressionData) == 0 {
		return fmt.Errorf("empty compression data")
	}

	// Check if first item has compression size specific fields
	firstItem := compressionData[0]
	if firstItem.CompressionType == "" || firstItem.UncompressedSize == 0 || firstItem.CompressedSize == 0 {
		return fmt.Errorf("not compression size data")
	}

	// Reset compression metrics
	compressionMetrics := []string{"compression_ratio", "index_size_bytes", "uncompressed_size_bytes", "memory_usage_mb"}
	for _, metricName := range compressionMetrics {
		if metric := e.metrics[metricName]; metric != nil {
			metric.Reset()
		}
	}

	// Extract dataset name from filepath for labeling
	dataset := extractDatasetName(filepath)

	// Update compression metrics
	for _, data := range compressionData {
		labels := prometheus.Labels{
			"compression_type": data.CompressionType,
			"dataset":         dataset,
		}

		if metric := e.metrics["compression_ratio"]; metric != nil {
			metric.With(labels).Set(data.CompressionRatio)
		}
		if metric := e.metrics["index_size_bytes"]; metric != nil {
			metric.With(labels).Set(float64(data.CompressedSize))
		}
		if metric := e.metrics["uncompressed_size_bytes"]; metric != nil {
			metric.With(labels).Set(float64(data.UncompressedSize))
		}
		if metric := e.metrics["memory_usage_mb"]; metric != nil {
			metric.With(labels).Set(data.MemoryUsageMB)
		}
	}

	log.Printf("Successfully processed compression size file: %s", filepath)
	return nil
}

func (e *Exporter) processCompressionRecallFile(content []byte, filepath string) error {
	var compressionData []CompressionRecallInfo
	if err := json.Unmarshal(content, &compressionData); err != nil {
		return err
	}

	// Validate that this is actually compression recall data
	if len(compressionData) == 0 {
		return fmt.Errorf("empty compression data")
	}

/*
	// Check if first item has compression recall specific fields
	firstItem := compressionData[0]
	if firstItem.CompressionType == "" || (firstItem.Recall == 0 && firstItem.QueriesPerSecond == 0) {
		return fmt.Errorf("not compression recall data")
	}
 */

	// Reset compression recall metrics
	compressionMetrics := []string{"recall_by_compression", "qps_by_compression", "latency_by_compression"}
	for _, metricName := range compressionMetrics {
		if metric := e.metrics[metricName]; metric != nil {
			metric.Reset()
		}
	}

	// Extract dataset name from filepath for labeling
	dataset := extractDatasetName(filepath)

	// Update compression recall metrics
	for _, data := range compressionData {
		labels := prometheus.Labels{
			"compression_type": data.CompressionType,
			"dataset":         dataset,
		}

		if metric := e.metrics["recall_by_compression"]; metric != nil {
			metric.With(labels).Set(data.Recall)
		}
		if metric := e.metrics["qps_by_compression"]; metric != nil {
			metric.With(labels).Set(data.QueriesPerSecond)
		}
		if metric := e.metrics["latency_by_compression"]; metric != nil {
			metric.With(labels).Set(data.MeanLatency)
		}
	}

	log.Printf("Successfully processed compression recall file: %s", filepath)
	return nil
}

func (e *Exporter) processRegularMetricsFile(content []byte, filepath string) error {
	var metricsData []MetricData
	if err := json.Unmarshal(content, &metricsData); err != nil {
		return fmt.Errorf("error parsing JSON from file %s: %v", filepath, err)
	}

	// Reset metrics before processing new data
	for _, metric := range e.metrics {
		// Only reset regular metrics, not compression metrics
		if !isCompressionMetric(metric) {
			metric.Reset()
		}
	}

	// Update metrics with new values
	for _, data := range metricsData {
		if data.Branch == "" {
			data.Branch = "main"
		}

		if data.TestID == "" {
			data.TestID = "NA"
		}

		labels := prometheus.Labels{
			"branch":          data.Branch,
			"dataset":         data.DatasetFile,
			"ef_construction": fmt.Sprintf("%d", data.EFConstruction),
			"max_connections": fmt.Sprintf("%d", data.MaxConnections),
			"limit":           fmt.Sprintf("%d", data.Limit),
			"ef":              fmt.Sprintf("%d", data.EF),
			"shards":          fmt.Sprintf("%d", data.Shards),
			"run_id":         data.TestID,
		}

		if metric := e.metrics["latency_mean"]; metric != nil {
			metric.With(labels).Set(data.MeanLatency)
		}
		if metric := e.metrics["latency_p99"]; metric != nil {
			metric.With(labels).Set(data.P99Latency)
		}
		if metric := e.metrics["qps"]; metric != nil {
			metric.With(labels).Set(data.QPS)
		}
		if metric := e.metrics["recall"]; metric != nil {
			metric.With(labels).Set(data.Recall)
		}
		if metric := e.metrics["import_time"]; metric != nil {
			metric.With(labels).Set(data.ImportTime)
		}
		if metric := e.metrics["heap_inuse_bytes"]; metric != nil {
			metric.With(labels).Set(data.HeapInuseBytes)
		}
		if metric := e.metrics["heap_alloc_bytes"]; metric != nil {
			metric.With(labels).Set(data.HeapAllocBytes)
		}
		if metric := e.metrics["heap_sys_bytes"]; metric != nil {
			metric.With(labels).Set(data.HeapSysBytes)
		}
	}

	log.Printf("Successfully processed regular metrics file: %s", filepath)
	return nil
}

func extractDatasetName(filepath string) string {
	// Extract dataset name from filepath
	// For files like "compression_analysis_compression_test_1756714286.json"
	// Extract "compression_test" as dataset name
	baseName := filepath
	if lastSlash := strings.LastIndex(filepath, "/"); lastSlash != -1 {
		baseName = filepath[lastSlash+1:]
	}

	// Remove .json extension
	if strings.HasSuffix(baseName, ".json") {
		baseName = baseName[:len(baseName)-5]
	}

	// For compression analysis files, try to extract meaningful dataset name
	if strings.HasPrefix(baseName, "compression_analysis_") {
		parts := strings.Split(baseName, "_")
		if len(parts) >= 4 {
			// Return the part between "compression_analysis_" and timestamp
			datasetParts := parts[2 : len(parts)-1] // Skip "compression", "analysis" and timestamp
			return strings.Join(datasetParts, "_")
		}
	}

	return baseName
}

func isCompressionMetric(metric *prometheus.GaugeVec) bool {
	// This is a simple heuristic - in a real implementation you might want to track this more explicitly
	return false // For now, we don't reset any metrics to avoid conflicts
}

func findLatestJSONFile(dirPath string) (string, error) {
	var latestFile string
	var latestTime time.Time

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if filepath.Ext(path) == ".json" && !info.IsDir() {
			if info.ModTime().After(latestTime) {
				latestTime = info.ModTime()
				latestFile = path
			}
		}
		return nil
	})

	if err != nil {
		return "", fmt.Errorf("error walking directory: %v", err)
	}

	if latestFile == "" {
		return "", fmt.Errorf("awaiting results")
	}

	return latestFile, nil
}

func pollDirectory(dirPath string, exporter *Exporter) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	var lastProcessedFile string

	for range ticker.C {
		latestFile, err := findLatestJSONFile(dirPath)
		if err != nil {
			log.Printf("Unable to public metrics: %v", err)
			continue
		}

		// Only process if it's a new file or hasn't been processed yet
		if latestFile != lastProcessedFile {
			if err := exporter.processJSONFile(latestFile); err != nil {
				log.Printf("Error processing file %s: %v", latestFile, err)
				continue
			}
			lastProcessedFile = latestFile
		}
	}
}

func main() {
	var (
		dirPath string
		port    int
	)

	// Create root command
	rootCmd := &cobra.Command{
		Use:   "metrics-exporter",
		Short: "Performance Metrics Exporter",
		Long:  `Monitor weaviate performance metrics and export via Prometheus.`,
		Run: func(cmd *cobra.Command, args []string) {

			prometheus.Unregister(prometheus.NewGoCollector())
			exporter := NewExporter()
			exporter.initializeMetrics()

			// Start polling directory
			go pollDirectory(dirPath, exporter)

			// Set up HTTP server
			http.Handle("/metrics", promhttp.Handler())
			http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(`<html>
					<head><title>Performance Metrics Exporter</title></head>
					<body>
						<h1>Performance Metrics Exporter</h1>
						<p><a href="/metrics">Metrics</a></p>
					</body>
					</html>`))
			})

			// Start server
			serverAddr := fmt.Sprintf(":%d", port)
			log.Printf("Starting metrics server on port %s", serverAddr)
			if err := http.ListenAndServe(serverAddr, nil); err != nil {
				log.Fatal(err)
			}
		},
		PreRunE: func(cmd *cobra.Command, args []string) error {
			// Validate required arguments
			if dirPath == "" {
				return fmt.Errorf("directory path is required")
			}
			return nil
		},
	}

	rootCmd.Flags().StringVarP(&dirPath, "dir", "d", "", "Results directory path to watch (required)")
	rootCmd.MarkFlagRequired("dir")
	rootCmd.Flags().IntVarP(&port, "port", "p", 2120, "Port to serve metrics on")

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
