"""Kubernetes deployment for OpenMythos.

Generates K8s manifests for production deployment.
"""
import os
import json
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class K8sBuilder:
    """Build Kubernetes manifests for OpenMythos.
    
    Args:
        namespace: K8s namespace
        replicas: Number of replicas
        model_size: Model variant
        gpu_enabled: Enable GPU nodes
    """
    namespace: str = "openmythos"
    replicas: int = 2
    model_size: str = "1b"
    gpu_enabled: bool = True
    output_dir: str = "deployment/k8s"

    def generate_namespace(self) -> Dict:
        """Generate namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": self.namespace},
        }

    def generate_deployment(self) -> Dict:
        """Generate Deployment manifest."""
        container = {
            "name": "openmythos",
            "image": f"openmythos:{self.model_size}",
            "ports": [{"containerPort": 8000}],
            "env": [
                {"name": "MODEL_SIZE", "value": self.model_size},
                {"name": "QUANTIZATION", "value": "int4"},
                {"name": "DEVICE", "value": "cuda" if self.gpu_enabled else "cpu"},
            ],
            "resources": {
                "requests": {"memory": "4Gi", "cpu": "2"},
                "limits": {"memory": "16Gi", "cpu": "4"},
            },
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": 8000},
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
            },
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": 8000},
                "initialDelaySeconds": 10,
                "periodSeconds": 5,
            },
        }

        if self.gpu_enabled:
            container["resources"]["limits"]["nvidia.com/gpu"] = "1"
            container["nodeSelector"] = {"gpu": "true"}

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "openmythos",
                "namespace": self.namespace,
                "labels": {"app": "openmythos"},
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {"matchLabels": {"app": "openmythos"}},
                "template": {
                    "metadata": {"labels": {"app": "openmythos"}},
                    "spec": {"containers": [container]},
                },
            },
        }

    def generate_service(self) -> Dict:
        """Generate Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "openmythos",
                "namespace": self.namespace,
            },
            "spec": {
                "selector": {"app": "openmythos"},
                "ports": [{"port": 80, "targetPort": 8000}],
                "type": "ClusterIP",
            },
        }

    def generate_hpa(self, min_replicas: int = 1, max_replicas: int = 10,
                     target_cpu: int = 70) -> Dict:
        """Generate HorizontalPodAutoscaler manifest."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "openmythos",
                "namespace": self.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "openmythos",
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": target_cpu},
                    },
                }],
            },
        }

    def generate_ingress(self, host: str = "openmythos.example.com") -> Dict:
        """Generate Ingress manifest."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "openmythos",
                "namespace": self.namespace,
                "annotations": {
                    "nginx.ingress.kubernetes.io/proxy-body-size": "50m",
                },
            },
            "spec": {
                "rules": [{
                    "host": host,
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": "openmythos",
                                    "port": {"number": 80},
                                },
                            },
                        }],
                    },
                }],
            },
        }

    def build(self, host: str = "openmythos.example.com"):
        """Generate all K8s manifests."""
        os.makedirs(self.output_dir, exist_ok=True)

        manifests = {
            "namespace.yaml": self.generate_namespace(),
            "deployment.yaml": self.generate_deployment(),
            "service.yaml": self.generate_service(),
            "hpa.yaml": self.generate_hpa(),
            "ingress.yaml": self.generate_ingress(host),
        }

        for filename, manifest in manifests.items():
            path = os.path.join(self.output_dir, filename)
            with open(path, "w") as f:
                import yaml
                yaml.dump(manifest, f, default_flow_style=False)

        return self.output_dir
