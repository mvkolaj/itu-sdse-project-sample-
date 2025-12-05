package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

// This struct *is your module*
type MLOpsPipeline struct{}

// Train runs your full Python pipeline inside Dagger
func (m *MLOpsPipeline) Train(ctx context.Context) (string, error) {

	// Connect to Dagger engine
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(nil))
	if err != nil {
		return "", err
	}
	defer client.Close()

	// Build Python environment container
	container := client.Container().
		From("python:3.10").
		WithMountedDirectory("/src", client.Host().Directory(".")).
		WithWorkdir("/src").
		WithExec([]string{"pip", "install", "-r", "notebooks/requirements.txt"}).
		WithExec([]string{"python", "main.py"})

	// Capture output
	out, err := container.Stdout(ctx)
	if err != nil {
		return "", err
	}

	return out, nil
}

// Required: Dagger entrypoint
func main() {
	ctx := context.Background()

	// Instantiate module
	pipeline := &MLOpsPipeline{}

	// Run training
	result, err := pipeline.Train(ctx)
	if err != nil {
		panic(err)
	}

	fmt.Println(result)
}
