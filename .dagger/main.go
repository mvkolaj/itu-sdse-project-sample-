package main

import (
	"context"

	"dagger/mlops-pipeline/internal/dagger"
)

type MlopsPipeline struct{}

// Train runs the full Python MLOps pipeline inside Dagger
func (m *MlopsPipeline) Train(ctx context.Context) (string, error) {

	ctr := dag.Container().
		From("python:3.10").
		WithMountedDirectory("/src", dag.Host().Directory(".")).
		WithWorkdir("/src").
		WithExec([]string{"pip", "install", "-r", "notebooks/requirements.txt"}).
		WithExec([]string{"python", "main.py"})

	return ctr.Stdout(ctx)
}

// Dagger automatically creates a CLI for this module
func main() {
	dagger.Serve(&MlopsPipeline{})
}
