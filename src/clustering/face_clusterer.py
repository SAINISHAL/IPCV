"""Unsupervised Face Clustering Module (DBSCAN & Agglomerative with Cosine Distance)."""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from src.utils.logger import setup_logger

logger = setup_logger("FaceClusterer")


class FaceClusterer:
    """Clusters facial embeddings without assuming student identity or fixed cluster counts."""

    def __init__(self, distance_threshold: float = 0.38, min_cluster_size: int = 3):
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.embeddings: Optional[np.ndarray] = None
        self.face_metadata: List[Dict[str, Any]] = []
        self.labels: Optional[np.ndarray] = None
        self.clusters: Dict[int, List[Dict[str, Any]]] = {}

    def cluster_faces(self, face_metadata: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Perform clustering on face metadata containing 'embedding'.
        
        Returns:
            Dictionary mapping cluster_id (int) -> list of face items.
            Cluster -1 denotes unassigned / noise / uncertain faces.
        """
        self.face_metadata = face_metadata
        if not face_metadata:
            return {}

        embeddings = np.array([item["embedding"] for item in face_metadata], dtype=np.float32)
        # Ensure L2 normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = embeddings / norms

        logger.info(f"Computing cosine distance matrix for {len(self.embeddings)} detected faces...")
        dist_matrix = cosine_distances(self.embeddings)

        # Use DBSCAN for density-based discovery of clusters and automatic noise detection
        logger.info(f"Running DBSCAN (eps={self.distance_threshold}, min_samples={self.min_cluster_size})...")
        db = DBSCAN(eps=self.distance_threshold, min_samples=self.min_cluster_size, metric="precomputed")
        self.labels = db.fit_predict(dist_matrix)

        # Organize by cluster ID
        self.clusters = {}
        for idx, lbl in enumerate(self.labels):
            cid = int(lbl)
            if cid not in self.clusters:
                self.clusters[cid] = []
            item = dict(self.face_metadata[idx])
            item["cluster_id"] = cid
            self.clusters[cid].append(item)

        n_clusters = len([c for c in self.clusters.keys() if c >= 0])
        n_noise = len(self.clusters.get(-1, []))
        logger.info(f"Clustering completed: {n_clusters} person clusters discovered, {n_noise} uncertain/noise faces.")

        return self.clusters

    def get_cluster_summaries(self) -> List[Dict[str, Any]]:
        """Calculate statistics, average similarity, and exemplar face for each cluster."""
        summaries = []
        for cid, items in sorted(self.clusters.items()):
            n_items = len(items)
            c_embeddings = np.array([it["embedding"] for it in items], dtype=np.float32)

            if cid >= 0 and n_items > 1:
                centroid = np.mean(c_embeddings, axis=0, keepdims=True)
                centroid /= np.linalg.norm(centroid)
                sims = np.dot(c_embeddings, centroid.T).flatten()
                avg_sim = float(np.mean(sims))
                exemplar_idx = int(np.argmax(sims))
                exemplar_face_id = items[exemplar_idx]["face_id"]
            else:
                avg_sim = 0.0
                exemplar_face_id = items[0]["face_id"] if n_items > 0 else "N/A"

            summaries.append({
                "cluster_id": cid,
                "is_noise": (cid == -1),
                "count": n_items,
                "avg_cosine_similarity": round(avg_sim, 3),
                "exemplar_face_id": exemplar_face_id,
                "face_ids": [it["face_id"] for it in items]
            })
        return summaries

    def save_clusters(self, filepath: str or Path) -> None:
        """Save clusters and metadata to JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "distance_threshold": self.distance_threshold,
            "min_cluster_size": self.min_cluster_size,
            "summaries": self.get_cluster_summaries(),
            "clusters": {
                str(cid): [
                    {k: v for k, v in item.items() if k != "embedding"}
                    for item in items
                ]
                for cid, items in self.clusters.items()
            }
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Clusters saved to {p}")

