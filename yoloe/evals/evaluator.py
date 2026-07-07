"""Evaluation framework for benchmarking detection accuracy."""

import json
import numpy as np
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """Results from evaluation."""
    method: str
    page: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    per_class_results: Dict[str, Dict[str, float]]


class Evaluator:
    """Evaluation framework for icon detection methods."""
    
    def __init__(self, ground_truth_config: str):
        """
        Initialize evaluator with ground truth configuration.
        
        Args:
            ground_truth_config: Path to JSON file containing ground truth data
        """
        self.ground_truth_config = ground_truth_config
        self.ground_truth = self.load_ground_truth()
    
    def load_ground_truth(self) -> Dict:
        """Load ground truth data from configuration file."""
        config_path = Path(self.ground_truth_config)
        if not config_path.exists():
            # Create a template ground truth file
            self.create_ground_truth_template()
            raise FileNotFoundError(f"Ground truth config not found. Created template at {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def create_ground_truth_template(self):
        """Create a template ground truth configuration file."""
        template = {
            "description": "Ground truth configuration for icon detection evaluation",
            "version": "1.0",
            "pages": {
                "page2.png": {
                    "expected_counts": {
                        "fan": 0,
                        "10amp-socket": 0,
                        "15amp-socket": 0,
                        "dbl-10amp-usb-socket": 0,
                        "perm-connect-isolator": 0
                    },
                    "detailed_annotations": [
                        {
                            "icon_name": "example-icon",
                            "x": 100,
                            "y": 200,
                            "width": 50,
                            "height": 50,
                            "rotation": 0,
                            "confidence": 1.0
                        }
                    ]
                },
                "Screenshot 2025-09-04 at 10.05.25.png": {
                    "expected_counts": {
                        "fan": 0,
                        "10amp-socket": 0,
                        "15amp-socket": 0,
                        "dbl-10amp-usb-socket": 0,
                        "perm-connect-isolator": 0
                    },
                    "detailed_annotations": []
                }
            },
            "evaluation_settings": {
                "iou_threshold": 0.5,
                "confidence_threshold": 0.5,
                "max_detection_distance": 20
            }
        }
        
        config_path = Path(self.ground_truth_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Created ground truth template at {config_path}")
        print("Please edit this file with the correct ground truth data for your pages.")
    
    def evaluate_method(self, detection_results: Dict[str, Dict[str, int]], 
                       method_name: str) -> List[EvaluationResult]:
        """
        Evaluate a detection method against ground truth.
        
        Args:
            detection_results: Results from detection method {page_name: {icon_name: count}}
            method_name: Name of the detection method
            
        Returns:
            List of evaluation results for each page
        """
        results = []
        
        for page_name, detected_counts in detection_results.items():
            if page_name not in self.ground_truth['pages']:
                print(f"Warning: No ground truth data for page {page_name}")
                continue
            
            gt_data = self.ground_truth['pages'][page_name]
            expected_counts = gt_data['expected_counts']
            
            # Calculate metrics
            eval_result = self._calculate_metrics(
                detected_counts, expected_counts, method_name, page_name
            )
            results.append(eval_result)
        
        return results
    
    def _calculate_metrics(self, detected: Dict[str, int], expected: Dict[str, int],
                          method: str, page: str) -> EvaluationResult:
        """Calculate evaluation metrics for a single page."""
        # Ensure all icon classes are represented
        all_classes = set(detected.keys()) | set(expected.keys())
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        per_class_results = {}
        
        for icon_class in all_classes:
            detected_count = detected.get(icon_class, 0)
            expected_count = expected.get(icon_class, 0)
            
            # Calculate TP, FP, FN for this class
            tp = min(detected_count, expected_count)
            fp = max(0, detected_count - expected_count)
            fn = max(0, expected_count - detected_count)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Calculate per-class metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if tp == 0 and fn == 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_results[icon_class] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'detected_count': detected_count,
                'expected_count': expected_count
            }
        
        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0 if total_tp == 0 and total_fn == 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy based on correct counts
        total_expected = sum(expected.values())
        total_detected = sum(detected.values())
        accuracy = 1.0 - abs(total_detected - total_expected) / max(total_expected, 1)
        accuracy = max(0.0, accuracy)  # Ensure non-negative
        
        return EvaluationResult(
            method=method,
            page=page,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            per_class_results=per_class_results
        )
    
    def compare_methods(self, results: Dict[str, List[EvaluationResult]]) -> Dict:
        """
        Compare multiple detection methods.
        
        Args:
            results: Dictionary of method_name -> evaluation_results
            
        Returns:
            Comparison summary
        """
        comparison = {
            'method_summary': {},
            'per_page_comparison': {},
            'overall_ranking': []
        }
        
        # Calculate method summaries
        for method_name, method_results in results.items():
            if not method_results:
                continue
            
            avg_precision = np.mean([r.precision for r in method_results])
            avg_recall = np.mean([r.recall for r in method_results])
            avg_f1 = np.mean([r.f1_score for r in method_results])
            avg_accuracy = np.mean([r.accuracy for r in method_results])
            
            comparison['method_summary'][method_name] = {
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1_score': avg_f1,
                'avg_accuracy': avg_accuracy,
                'total_pages': len(method_results)
            }
        
        # Per-page comparison
        all_pages = set()
        for method_results in results.values():
            all_pages.update(r.page for r in method_results)
        
        for page in all_pages:
            comparison['per_page_comparison'][page] = {}
            for method_name, method_results in results.items():
                page_results = [r for r in method_results if r.page == page]
                if page_results:
                    result = page_results[0]  # Should be only one per page
                    comparison['per_page_comparison'][page][method_name] = {
                        'precision': result.precision,
                        'recall': result.recall,
                        'f1_score': result.f1_score,
                        'accuracy': result.accuracy
                    }
        
        # Overall ranking by F1 score
        method_scores = [(name, summary['avg_f1_score']) 
                        for name, summary in comparison['method_summary'].items()]
        method_scores.sort(key=lambda x: x[1], reverse=True)
        comparison['overall_ranking'] = method_scores
        
        return comparison
    
    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save evaluation results to JSON file."""
        serializable_results = []
        for result in results:
            serializable_results.append({
                'method': result.method,
                'page': result.page,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'accuracy': result.accuracy,
                'true_positives': result.true_positives,
                'false_positives': result.false_positives,
                'false_negatives': result.false_negatives,
                'per_class_results': result.per_class_results
            })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {output_path}")
    
    def save_comparison(self, comparison: Dict, output_path: str):
        """Save method comparison to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Method comparison saved to {output_path}")
    
    def print_summary(self, results: List[EvaluationResult]):
        """Print a summary of evaluation results."""
        print(f"\nEvaluation Summary for {results[0].method if results else 'Unknown Method'}")
        print("=" * 60)
        
        if not results:
            print("No results to display.")
            return
        
        # Overall metrics
        avg_precision = np.mean([r.precision for r in results])
        avg_recall = np.mean([r.recall for r in results])
        avg_f1 = np.mean([r.f1_score for r in results])
        avg_accuracy = np.mean([r.accuracy for r in results])
        
        print(f"Overall Metrics:")
        print(f"  Precision: {avg_precision:.3f}")
        print(f"  Recall: {avg_recall:.3f}")
        print(f"  F1 Score: {avg_f1:.3f}")
        print(f"  Accuracy: {avg_accuracy:.3f}")
        print()
        
        # Per-page results
        print("Per-Page Results:")
        for result in results:
            print(f"  {result.page}:")
            print(f"    Precision: {result.precision:.3f}, Recall: {result.recall:.3f}, F1: {result.f1_score:.3f}")
            print(f"    TP: {result.true_positives}, FP: {result.false_positives}, FN: {result.false_negatives}")
        print()


def run_evaluation_test(ground_truth_config: str):
    """
    Run a complete evaluation test using the configuration file.
    
    Args:
        ground_truth_config: Path to ground truth configuration file
    """
    evaluator = Evaluator(ground_truth_config)
    
    # This is a placeholder - in actual usage, you would pass real detection results
    example_results = {
        "page2.png": {
            "fan": 2,
            "10amp-socket": 1,
            "15amp-socket": 0,
            "dbl-10amp-usb-socket": 1,
            "perm-connect-isolator": 0
        }
    }
    
    results = evaluator.evaluate_method(example_results, "example_method")
    evaluator.print_summary(results)
    
    return results