package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()
	if err := run(ctx); err != nil {
		log.Fatalf("Dagger pipeline failed: %v", err)
	}
	log.Println("Dagger pipeline completed successfully!")
}

func run(ctx context.Context) error {
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer()))
	if err != nil {
		return err
	}
	defer client.Close()

	// 🔽 REPLACE this string with your actual repo root from `pwd`
	hostRepo := client.Host().Directory("/Users/mikolajandrzejewski/Documents/GitHub/itu-sdse-project-sample")

	python := client.Container().
		From("python:3.10").
		WithMountedDirectory("/src", hostRepo).
		WithExec([]string{"ls", "-R", "/src"}).
		WithWorkdir("/src").
		WithExec([]string{"python", "-m", "pip", "install", "--upgrade", "pip"}).
		WithExec([]string{"pip", "install", "-r", "/src/requirements.txt"}).
		WithExec([]string{"python", "main.py"})

	_, err = python.Sync(ctx)
	return err
}
