package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"log"
	"math"
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
	outputFile        = ""
	inputFile         = ""
	extension         = ".jpg"
	extractPeaks      = true
	embeddingFile     = ""
	clusters          = ""
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
	flag.StringVar(&clusters, "clusters", clusters, "cluster output directory")
}

func l2(a, b []float32) float32 {
	sum := 0.
	for i := 0; i < len(a); i++ {
		d := b[i] - a[i]
		sum += float64(d * d)
	}
	return float32(math.Sqrt(sum))
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

func fileCopyNew(from, to string) error {
	f, err := os.Open(from)
	if err != nil {
		return err
	}
	defer f.Close()
	t, err := os.OpenFile(to, os.O_CREATE|os.O_WRONLY, 0660)
	if err != nil {
		return err
	}
	defer t.Close()

	// buf := make([]byte, 65535)

	// for {
	// 	n, err := f.Read(buf)
	// 	if err == io.EOF || err == nil {
	// 		t.Write(buf[0:n])
	// 	}
	// 	if err == io.EOF {
	// 		break
	// 	}
	// }

	b, err := ioutil.ReadAll(f)
	if err != nil {
		return err
	}
	if n, err := t.Write(b); err != nil {
		return err
	} else if n < len(b) {
		return errors.New("not a complete file written")
	}
	return nil
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
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
			if !os.IsNotExist(err) {
				log.Fatal(err)
			}
		} else {
			b, err := ioutil.ReadAll(f)
			f.Close()

			if err != nil {
				log.Fatal(err)
			} else if err := json.Unmarshal(b, &embeddings); err != nil {
				log.Fatal(err)
			}
		}
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
		b, err := json.Marshal(embeddings)
		if err != nil {
			log.Fatal(err)
		}

		f, err := os.OpenFile(embeddingFile, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0660)
		if err != nil {
			log.Fatal(err)
		}

		_, err = f.Write(b)
		f.Close()

		if err != nil {
			log.Fatal(err)
		}
	}

	peaks := multiModal.Peaks()

	if extractPeaks {
		b, err := json.MarshalIndent(peaks, "", "  ")
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

	count := 0

	if clusters != "" && len(peaks) > 0 {
		if err := os.Mkdir(clusters, 0770); err != nil {
			log.Fatal(err)
		}

		for p, e := range embeddings {
			if count%100 == 0 {
				fmt.Printf("cluster: %d", count)
			}
			min_distance := float32(math.MaxFloat32)
			min_peak := -1

			for i, peak := range peaks {
				dist := l2(peak.Mean, e)

				if dist < min_distance {
					min_distance = dist
					min_peak = i
				}
			}

			// copy file from path p to directory clusters
			if err := os.MkdirAll(fmt.Sprintf("%s%s%d", clusters, os.PathSeparator, peaks[min_peak].Id), 0770); err != nil {
				log.Fatal(err)
			}

			if err := fileCopyNew(p, fmt.Sprintf("%s%s%d%simage%d%s", clusters, os.PathSeparator, peaks[min_peak].Id, count, extension)); err != nil {
				log.Fatal(err)
			}
			count++
		}
	}
}
