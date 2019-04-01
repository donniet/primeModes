package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"image/jpeg"
	"log"
	"os"
	"path/filepath"

	detector "github.com/donniet/detector"
)

var (
	facesDirectory    = ""
	recurse           = true
	classifierDesc    = ""
	classifierWeights = ""
	device            = "CPU"
	maximumNodes      = 1024
	outputFile        = "output.multimodal"
	inputFile         = ""
	extension         = ".jpg"
	extractPeaks      = true
	embeddingFile     = ""
)

func init() {
	flag.StringVar(&facesDirectory, "faces", facesDirectory, "root directory with prime faces")
	flag.BoolVar(&recurse, "recurse", recurse, "recurse subdirectories")
	flag.BoolVar(&recurse, "r", recurse, "recurse subdirectories")
	flag.StringVar(&classifierDesc, "classifierDesc", classifierDesc, "classifier description file")
	flag.StringVar(&classifierWeights, "classifierWeights", classifierWeights, "classifier weights file")
	flag.StringVar(&device, "device", device, "device to run classification on")
	flag.IntVar(&maximumNodes, "nodes", maximumNodes, "maximum number of nodes in modal structure")
	flag.StringVar(&outputFile, "output", outputFile, "output file for modal structure")
	flag.StringVar(&outputFile, "o", outputFile, "output file for modal structure")
	flag.StringVar(&inputFile, "input", inputFile, "input modal to start from (blank for none)")
	flag.StringVar(&inputFile, "i", inputFile, "input modal to start from (blank for none)")
	flag.StringVar(&extension, "extension", extension, "image extension (only .jpg supported)")
	flag.BoolVar(&extractPeaks, "peaks", extractPeaks, "extract the peaks from the data structure")
	flag.StringVar(&embeddingFile, "embeddings", embeddingFile, "read or write the embeddings to this file")
}

func processImage(path string, classer *detector.Classifier, multiModal *detector.MultiModal) ([]float32, error) {
	var rgb *detector.RGB24

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	if img, err := jpeg.Decode(f); err != nil {
		return nil, err
	} else {
		rgb = detector.FromImage(img)
	}

	res := classer.InferRGB24(rgb)

	multiModal.Insert(res.Embedding)
	return res.Embedding, nil
}

func main() {
	flag.Parse()

	var classer *detector.Classifier
	embeddingSize := 128

	if classifierDesc != "" {
		classer = detector.NewClassifier(classifierDesc, classifierWeights, device)
		defer classer.Close()
		embeddingSize = classer.EmbeddingSize()
	}

	multiModal := detector.NewMultiModal(embeddingSize, maximumNodes)
	defer multiModal.Close()

	if inputFile == "" {
		// do nothing
	} else if f, err := os.OpenFile(inputFile, os.O_RDONLY, 0660); err != nil {
		log.Fatal(err)
	} else if _, err := multiModal.ReadFrom(f); err != nil {
		f.Close()
		log.Fatal(err)
	} else {
		f.Close()
	}

	// go through all files in the directory
	images := []string{}

	if facesDirectory != "" {
		err := filepath.Walk(facesDirectory, func(path string, f os.FileInfo, err error) error {
			if filepath.Ext(path) == extension {
				images = append(images, path)
			}
			return nil
		})
		if err != nil {
			log.Fatal(err)
		}
	}

	embeddings := make(map[string][]float32)

	if embeddingFile != "" {
		f, err := os.Open(embeddingFile)
		if err != nil {
			log.Fatal(err)
		}
		if err := binary.Read(f, binary.LittleEndian, &embeddingFile); err != nil {
			f.Close()
			log.Fatal(err)
		}
		f.Close()
	}

	// for each image, calculate the embedding
	for i, p := range images {
		if i%100 == 0 {
			log.Printf("image #%d", i)
		}

		if embedding, err := processImage(p, classer, &multiModal); err != nil {
			log.Print(err)
		} else {
			embeddings[p] = embedding
		}
	}

	if embeddingFile != "" {
		f, err := os.OpenFile(embeddingFile, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0660)
		if err != nil {
			log.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, embeddings); err != nil {
			f.Close()
			log.Fatal(f)
		}
		f.Close()
	}

	if extractPeaks {
		b, err := json.MarshalIndent(multiModal.Peaks(), "", "  ")
		if err != nil {
			log.Fatal(err)
		}
		os.Stdout.Write(b)
	}

	if outputFile != "" {
		log.Printf("writing file...")
		f, err := os.OpenFile(outputFile, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0660)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		if _, err := multiModal.WriteTo(f); err != nil {
			log.Fatal(err)
		}
	}
}
