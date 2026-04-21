import click
import json
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from tqdm import tqdm


class NNIndex:
    """Encapsulates the vector data and search logic."""

    def __init__(self, id_key: str = 'cust_id'):
        """
        Initializes the index.
        Args:
            id_key (str): The key to use for the identifier in the JSONL file.
        """
        self.id_key = id_key
        self.ids: List[Any] = []
        self.matrix_normalized: np.ndarray = None
        self.dimension: int = None
        print(f"INFO:     NNIndex configured to use '{self.id_key}' as the identifier key.")

    def load_from_jsonl(self, filepath: str):
        """
        Loads and normalizes vectors from a JSONL file.
        Infers vector dimension from the first entry.
        Shows a progress bar using tqdm.
        """
        vectors = []
        print(f"INFO:     Counting vectors in '{filepath}'...")
        with open(filepath, 'r') as f:
            num_lines = sum(1 for _ in f)

        print(f"INFO:     Loading and processing {num_lines} vectors...")
        with open(filepath, 'r') as f:
            for line in tqdm(f, total=num_lines, desc="Loading vectors"):
                data = json.loads(line)
                
                if self.id_key not in data:
                    raise KeyError(f"The specified id_key '{self.id_key}' was not found in a line in the JSONL file.")

                if "v" not in data:
                     raise KeyError(f"The vector key 'v' was not found in a line in the JSONL file.")

                vector_data = data["v"]
                id_data = data[self.id_key]

                if self.dimension is None:
                    self.dimension = len(vector_data)
                    print(f"INFO:     Inferred vector dimension: {self.dimension}")

                if len(vector_data) != self.dimension:
                    raise ValueError(
                        f"Inconsistent vector dimension in {filepath}. "
                        f"Expected {self.dimension}, got {len(vector_data)} for id {id_data}."
                    )
                
                self.ids.append(id_data)
                vectors.append(vector_data)

        if not vectors:
            print("WARNING:  No vectors were loaded.")
            self.matrix_normalized = np.array([], dtype=np.float32).reshape(0, self.dimension or 1)
            return

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.matrix_normalized = matrix / norms
        print("INFO:     Vector normalization complete.")


    def find_neighbors(self, query_vector: List[float], k: int) -> List[Dict]:
        """Finds the top-k nearest neighbors for a query vector."""
        if self.matrix_normalized is None or self.dimension is None:
            raise RuntimeError("Index is not loaded.")

        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector has incorrect dimension. Expected {self.dimension}, got {len(query_vector)}.")
        
        query_np = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query_np)
        if query_norm == 0:
            raise ValueError("Query vector cannot be a zero vector.")
        query_normalized = query_np / query_norm

        scores = self.matrix_normalized @ query_normalized
        effective_k = min(k, self.matrix_normalized.shape[0])
        
        if effective_k == 0:
            return []
            
        partitioned_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        top_k_indices = partitioned_indices[np.argsort(scores[partitioned_indices])[::-1]]

        return [
            {self.id_key: self.ids[idx], "score": float(scores[idx])}
            for idx in top_k_indices
        ]

class QueryVector(BaseModel):
    vector: List[float]


class NeighborsResponse(BaseModel):
    neighbors: List[Dict[str, Any]]


# --- FastAPI Application Factory ---
def create_app(vector_file: str, id_key: str) -> FastAPI:
    """Creates and configures a FastAPI application instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("INFO:     Application startup...")
        index = NNIndex(id_key=id_key)
        index.load_from_jsonl(vector_file)
        app.state.nn_index = index # Attach index to app state
        print(f"INFO:     Index loaded. Ready to serve {len(index.ids)} vectors.")
    
        yield
        
        # On shutdown
        print("INFO:     Application shutdown.")

    app = FastAPI(
        title="CLI-Powered Nearest Neighbor API",
        description="An API for finding nearest neighbors, configured via the command line.",
        version="0.4.0",
        lifespan=lifespan
    )

    @app.post("/neighbors", response_model=NeighborsResponse)
    async def get_neighbors(
        request: Request,
        query: QueryVector,
        k: int = 10,
    ):
        nn_index = request.app.state.nn_index
        if not nn_index or not nn_index.matrix_normalized is not None:
            raise HTTPException(status_code=503, detail="Vector index is not available.")

        results = nn_index.find_neighbors(query.vector, k)
        return NeighborsResponse(neighbors=results)

    @app.get("/", include_in_schema=False)
    def root():
        return {"message": "Nearest Neighbor API is running. POST to /neighbors"}

    return app

@click.command()
@click.option(
    '--vector-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the JSONL file containing the vectors."
)
@click.option(
    '--id-key',
    default='id',
    show_default=True,
    help="The key for the identifier in the JSONL file."
)
@click.option('--host', default='127.0.0.1', help='Host to bind the server to.')
@click.option('--port', default=8000, type=int, help='Port to run the server on.')
def main(vector_file: str, id_key: str, host: str, port: int):
    """
    Starts the Nearest Neighbor API server.
    """
    print(f"INFO:     Starting server with vector file: {vector_file}")
    app = create_app(vector_file=vector_file, id_key=id_key)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
