"""
Diagnostic utilities for CReSO package.

Provides system checks, compatibility tests, and troubleshooting guidance
to help users identify and resolve common issues.
"""

import sys
from typing import Dict, List, Any
import numpy as np
import torch

from .logging import get_logger
from .config import CReSOConfig

logger = get_logger(__name__)


class SystemDiagnostics:
    """System and environment diagnostics for CReSO."""

    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.info: List[Dict[str, Any]] = []

    def check_python_version(self) -> None:
        """Check if Python version is compatible."""
        min_version = (3, 9)
        current_version = sys.version_info[:2]

        if current_version < min_version:
            self.issues.append(
                {
                    "type": "python_version",
                    "severity": "error",
                    "message": f"Python {current_version[0]}.{current_version[1]} is not supported",
                    "recommendation": f"Please upgrade to Python {min_version[0]}.{min_version[1]} or later",
                    "current": f"{current_version[0]}.{current_version[1]}",
                    "required": f"{min_version[0]}.{min_version[1]}+",
                }
            )
        else:
            self.info.append(
                {
                    "type": "python_version",
                    "message": f"Python {current_version[0]}.{current_version[1]} is supported",
                    "status": "ok",
                }
            )

    def check_pytorch_version(self) -> None:
        """Check if PyTorch version is compatible."""
        try:
            torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
            min_version = (2, 0)

            if torch_version < min_version:
                self.issues.append(
                    {
                        "type": "pytorch_version",
                        "severity": "error",
                        "message": f"PyTorch {torch.__version__} is not supported",
                        "recommendation": f"Please upgrade to PyTorch {min_version[0]}.{min_version[1]} or later",
                        "current": torch.__version__,
                        "required": f"{min_version[0]}.{min_version[1]}+",
                    }
                )
            else:
                self.info.append(
                    {
                        "type": "pytorch_version",
                        "message": f"PyTorch {torch.__version__} is supported",
                        "status": "ok",
                    }
                )

        except Exception as e:
            self.issues.append(
                {
                    "type": "pytorch_version",
                    "severity": "error",
                    "message": f"Could not determine PyTorch version: {e}",
                    "recommendation": "Please ensure PyTorch is properly installed",
                }
            )

    def check_cuda_availability(self) -> None:
        """Check CUDA availability and configuration."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)

            self.info.append(
                {
                    "type": "cuda",
                    "message": f"CUDA is available with {gpu_count} GPU(s)",
                    "gpu_name": gpu_name,
                    "gpu_count": gpu_count,
                    "current_device": current_device,
                    "status": "ok",
                }
            )

            # Check memory
            try:
                torch.cuda.memory_allocated() / 1024**3  # GB
                torch.cuda.memory_reserved() / 1024**3  # GB
                total_memory = (
                    torch.cuda.get_device_properties(current_device).total_memory
                    / 1024**3
                )  # GB

                if total_memory < 2:  # Less than 2GB
                    self.warnings.append(
                        {
                            "type": "gpu_memory",
                            "message": f"GPU memory is limited ({total_memory:.1f} GB)",
                            "recommendation": "Consider using smaller batch sizes or models",
                            "memory_gb": total_memory,
                        }
                    )
                else:
                    self.info.append(
                        {
                            "type": "gpu_memory",
                            "message": f"GPU memory: {total_memory:.1f} GB available",
                            "memory_gb": total_memory,
                            "status": "ok",
                        }
                    )

            except Exception as e:
                self.warnings.append(
                    {
                        "type": "gpu_memory",
                        "message": f"Could not check GPU memory: {e}",
                        "recommendation": "GPU memory information unavailable",
                    }
                )
        else:
            self.info.append(
                {
                    "type": "cuda",
                    "message": "CUDA is not available, will use CPU",
                    "status": "cpu_only",
                }
            )

    def check_dependencies(self) -> None:
        """Check optional dependencies."""
        optional_deps = {
            "scipy": "Graph processing features",
            "matplotlib": "Visualization features",
            "seaborn": "Advanced plotting",
            "hydra-core": "Command line interface",
            "omegaconf": "Configuration management",
        }

        for dep_name, description in optional_deps.items():
            try:
                __import__(dep_name.replace("-", "_"))
                self.info.append(
                    {
                        "type": "dependency",
                        "message": f"{dep_name} is available ({description})",
                        "dependency": dep_name,
                        "status": "ok",
                    }
                )
            except ImportError:
                self.warnings.append(
                    {
                        "type": "dependency",
                        "message": f"{dep_name} is not installed",
                        "recommendation": f"Install with: pip install {dep_name}",
                        "feature": description,
                        "dependency": dep_name,
                    }
                )

    def check_memory_availability(self) -> None:
        """Check available system memory."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            available_gb = memory.available / 1024**3
            total_gb = memory.total / 1024**3

            if available_gb < 2:
                self.warnings.append(
                    {
                        "type": "system_memory",
                        "message": f"Low system memory: {available_gb:.1f} GB available",
                        "recommendation": "Consider using smaller datasets or batch sizes",
                        "available_gb": available_gb,
                        "total_gb": total_gb,
                    }
                )
            else:
                self.info.append(
                    {
                        "type": "system_memory",
                        "message": f"System memory: {available_gb:.1f} GB available of {total_gb:.1f} GB",
                        "available_gb": available_gb,
                        "total_gb": total_gb,
                        "status": "ok",
                    }
                )

        except ImportError:
            self.warnings.append(
                {
                    "type": "system_memory",
                    "message": "Cannot check system memory (psutil not available)",
                    "recommendation": "Install psutil for memory monitoring: pip install psutil",
                }
            )
        except Exception as e:
            self.warnings.append(
                {
                    "type": "system_memory",
                    "message": f"Error checking system memory: {e}",
                    "recommendation": "Memory information unavailable",
                }
            )

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all diagnostic checks."""
        logger.info("Running CReSO system diagnostics...")

        self.check_python_version()
        self.check_pytorch_version()
        self.check_cuda_availability()
        self.check_dependencies()
        self.check_memory_availability()

        return {
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
            "has_critical_issues": len(
                [i for i in self.issues if i.get("severity") == "error"]
            )
            > 0,
        }

    def print_report(self) -> None:
        """Print a human-readable diagnostic report."""
        results = self.run_all_checks()

        print("\n" + "=" * 60)
        print("CReSO SYSTEM DIAGNOSTICS")
        print("=" * 60)

        # Critical issues
        critical_issues = [i for i in results["issues"] if i.get("severity") == "error"]
        if critical_issues:
            print("\nCRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"   ERROR: {issue['message']}")
                if "recommendation" in issue:
                    print(f"      SUGGESTION: {issue['recommendation']}")

        # Warnings
        if results["warnings"]:
            print("\nWARNINGS:")
            for warning in results["warnings"]:
                print(f"   WARNING: {warning['message']}")
                if "recommendation" in warning:
                    print(f"      SUGGESTION: {warning['recommendation']}")

        # Info/OK status
        if results["info"]:
            print("\nSTATUS:")
            for info in results["info"]:
                print(f"   OK: {info['message']}")

        # Summary
        print("\nSUMMARY:")
        print(f"   • Critical Issues: {len(critical_issues)}")
        print(f"   • Warnings: {len(results['warnings'])}")
        print(f"   • OK Status: {len(results['info'])}")

        if critical_issues:
            print("\nCReSO may not work properly until critical issues are resolved.")
        elif results["warnings"]:
            print("\nCReSO should work, but some features may be limited.")
        else:
            print("\nAll checks passed! CReSO is ready to use.")

        print("=" * 60)


class ConfigurationValidator:
    """Validates and provides guidance for CReSOConfig objects."""

    @staticmethod
    def validate_config(config: CReSOConfig) -> Dict[str, List[str]]:
        """Validate a CReSOConfig and return issues/recommendations."""
        issues = []
        warnings = []
        recommendations = []

        # Check basic parameters
        if config.input_dim <= 0:
            issues.append("input_dim must be positive")
        elif config.input_dim > 10000:
            warnings.append(
                f"Large input dimension ({config.input_dim}) may require more memory"
            )

        if config.n_components <= 0:
            issues.append("n_components must be positive")
        elif config.n_components > config.input_dim:
            warnings.append(
                "More components than input dimensions may lead to overfitting"
            )

        # Check training parameters
        if config.lr <= 0 or config.lr > 1:
            issues.append(f"Learning rate ({config.lr}) should be in (0, 1]")
        elif config.lr > 0.1:
            warnings.append(
                f"High learning rate ({config.lr}) may cause training instability"
            )

        if config.epochs <= 0:
            issues.append("epochs must be positive")
        elif config.epochs > 1000:
            warnings.append(f"Many epochs ({config.epochs}) may lead to overfitting")

        if config.batch_size <= 0:
            issues.append("batch_size must be positive")
        elif config.batch_size < 32:
            warnings.append("Small batch size may lead to noisy gradients")
        elif config.batch_size > 2048:
            warnings.append("Large batch size may require more memory")

        # Check regularization
        if any(
            reg < 0 for reg in [config.l2_freq, config.group_l1, config.center_disp]
        ):
            issues.append("Regularization parameters must be non-negative")

        # Device checks
        if config.device == "cuda" and not torch.cuda.is_available():
            issues.append("CUDA device requested but not available")
            recommendations.append("Set device='cpu' or install CUDA")

        # Performance recommendations
        if config.use_amp and config.device == "cpu":
            warnings.append("AMP (Automatic Mixed Precision) is not beneficial on CPU")
            recommendations.append("Disable AMP for CPU training")

        if not config.use_seed_freqs:
            recommendations.append(
                "Consider enabling frequency seeding for better initialization"
            )

        return {
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    @staticmethod
    def print_config_validation(config: CReSOConfig) -> bool:
        """Print configuration validation results and return if valid."""
        results = ConfigurationValidator.validate_config(config)

        print("\n" + "=" * 50)
        print("CONFIGURATION VALIDATION")
        print("=" * 50)

        if results["issues"]:
            print("\nERROR: ISSUES:")
            for issue in results["issues"]:
                print(f"   • {issue}")

        if results["warnings"]:
            print("\nWARNINGS:")
            for warning in results["warnings"]:
                print(f"   • {warning}")

        if results["recommendations"]:
            print("\nSUGGESTION: RECOMMENDATIONS:")
            for rec in results["recommendations"]:
                print(f"   • {rec}")

        is_valid = len(results["issues"]) == 0

        if is_valid:
            print("\nOK: Configuration is valid!")
        else:
            print(
                f"\nERROR: Configuration has {len(results['issues'])} issue(s) that must be fixed."
            )

        print("=" * 50)
        return is_valid


def run_quick_test() -> bool:
    """Run a quick functionality test to verify CReSO is working."""
    try:
        print("\n" + "=" * 40)
        print("QUICK FUNCTIONALITY TEST")
        print("=" * 40)

        # Test basic imports
        print("Testing imports... ", end="")
        from .classifier import CReSOClassifier
        from .config import CReSOConfig

        print("OK")

        # Test basic model creation
        print("Testing model creation... ", end="")
        from .config import ModelArchitectureConfig, TrainingConfig, SystemConfig

        arch_config = ModelArchitectureConfig(input_dim=10, n_components=8)
        train_config = TrainingConfig(max_epochs=1)
        sys_config = SystemConfig(device="cpu")
        config = CReSOConfig(
            architecture=arch_config, training=train_config, system=sys_config
        )
        clf = CReSOClassifier(config)
        print("OK")

        # Test with small synthetic data
        print("Testing training... ", end="")
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        clf.fit(X, y)
        print("OK")

        # Test prediction
        print("Testing prediction... ", end="")
        clf.predict(X[:10])
        clf.predict_proba(X[:10])
        print("OK")

        print("\nQuick test passed! CReSO is working correctly.")
        print("=" * 40)
        return True

    except Exception as e:
        print(f"ERROR: Quick test failed: {e}")
        print("\nTroubleshooting suggestions:")
        print("  1. Check system diagnostics with: python -m creso.diagnostics")
        print("  2. Verify all dependencies are installed")
        print("  3. Try with CPU-only configuration")
        print("=" * 40)
        return False


def main():
    """Main function for running diagnostics from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="CReSO System Diagnostics")
    parser.add_argument(
        "--quick-test", action="store_true", help="Run quick functionality test"
    )
    parser.add_argument(
        "--config-check", action="store_true", help="Check default configuration"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full diagnostics (default)"
    )

    args = parser.parse_args()

    if args.quick_test:
        success = run_quick_test()
        sys.exit(0 if success else 1)

    if args.config_check:
        config = CReSOConfig()
        is_valid = ConfigurationValidator.print_config_validation(config)
        sys.exit(0 if is_valid else 1)

    # Default: run full diagnostics
    diagnostics = SystemDiagnostics()
    diagnostics.print_report()

    if args.quick_test or args.full:
        print("\n")
        run_quick_test()


if __name__ == "__main__":
    main()
