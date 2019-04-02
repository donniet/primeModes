package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	detector "github.com/donniet/detector"
)

var (
	modes = ""
)

func init() {
	flag.StringVar(&modes, "modal", modes, "modal input file")
}

func main() {
	flag.Parse()

	f, err := os.Open(modes)
	if err != nil {
		log.Fatal(err)
	}
	modal := detector.NewMultiModal(128, 0)

	_, err = modal.ReadFrom(f)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}

	peaks := modal.Peaks()
	b, _ := json.MarshalIndent(peaks, "", "  ")
	os.Stdout.Write(b)

	// log.Printf("dimensions: %d", modal.Dimensions())

	// fmt.Println("hello")
}
