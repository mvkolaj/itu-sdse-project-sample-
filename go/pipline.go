package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()
	if err := stage(ctx); err != nil {
		log.Fatalf("❌ Dagger pipeline failed: %v", err)
	}
	log.Println("✅ Dagger pipeline completed successfully!")
}

// stage runs the main ML pipeline
func stage(ctx context.Context) error {
	// 1️⃣ Create a Dagger client
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer()))
	if err != nil {
		return err
	}
	defer client.Close()

	// 2️⃣ Define the Python container for the ML pipeline
	pythonContainer := client.Container().
		From("python:3.10").
		// Mount repo root into /src in the container
		WithMountedDirectory("/src", client.Host().Directory(".")).
		WithWorkdir("/src").
		// Install Python dependencies (adjust path if your requirements live elsewhere)
		WithExec([]string{"pip", "install", "--upgrade", "pip"}).
		WithExec([]string{"pip", "install", "-r", "notebooks/requirements.txt"}).
		// Run your existing main.py
		WithExec([]string{"python", "main.py"})

	// 3️⃣ Sync outputs back to host (e.g., model/ folder)
	_, err = pythonContainer.Sync(ctx)
	return err
}
