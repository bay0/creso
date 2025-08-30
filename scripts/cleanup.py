#!/usr/bin/env python3
"""
Interactive cleanup tool for CReSO project.
Provides safe, configurable cleanup with dry-run and backup capabilities.
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class CReSOCleaner:
    """Interactive cleanup utility for CReSO project."""

    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / ".cleanrc"
        self.config = self.load_config()
        self.stats = {"files_deleted": 0, "dirs_deleted": 0, "space_freed": 0}

    def load_config(self) -> Dict:
        """Load cleanup configuration."""
        default_config = {
            "protected_patterns": [
                "examples/notebooks/*.ipynb",
                "docs/**/*",
                ".git/**/*",
                "*.md",
                "pyproject.toml",
                "setup.cfg",
                "Makefile",
                "LICENSE"
            ],
            "cleanup_patterns": {
                "cache": ["__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"],
                "python_compiled": ["*.pyc", "*.pyo"],
                "models": ["*.pkl", "*.pt", "*.ts", "*.onnx"],
                "logs": ["*.log", "*.out", "*.err"],
                "coverage": ["htmlcov", "coverage.xml", ".coverage*"],
                "results": ["*.png", "*.jpg", "*.csv", "results/", "benchmarks/results/"],
                "build": ["build/", "dist/", "*.egg-info/"]
            },
            "backup_dir": "cleanup_backups",
            "interactive": True,
            "dry_run": False
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration.")
        
        return default_config

    def save_config(self):
        """Save current configuration."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except IOError as e:
            print(f"Error saving config: {e}")

    def find_files(self, patterns: List[str]) -> List[Path]:
        """Find files matching patterns."""
        found_files = []
        
        for pattern in patterns:
            if pattern.endswith('/'):
                # Directory pattern
                for path in self.project_root.rglob(pattern.rstrip('/')):
                    if path.is_dir() and not self.is_protected(path):
                        found_files.append(path)
            else:
                # File pattern
                for path in self.project_root.rglob(pattern):
                    if path.is_file() and not self.is_protected(path):
                        found_files.append(path)
        
        return sorted(set(found_files))

    def is_protected(self, path: Path) -> bool:
        """Check if path is protected from cleanup."""
        relative_path = path.relative_to(self.project_root)
        
        for pattern in self.config["protected_patterns"]:
            if relative_path.match(pattern):
                return True
        
        return False

    def get_file_size(self, path: Path) -> int:
        """Get file or directory size in bytes."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total_size = 0
            try:
                for item in path.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size
            except (OSError, PermissionError):
                pass
            return total_size
        return 0

    def format_size(self, bytes_size: int) -> str:
        """Format bytes as human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"

    def backup_file(self, file_path: Path, backup_dir: Path):
        """Backup a file before deletion."""
        backup_dir.mkdir(parents=True, exist_ok=True)
        relative_path = file_path.relative_to(self.project_root)
        backup_path = backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.is_file():
            shutil.copy2(file_path, backup_path)
        elif file_path.is_dir():
            shutil.copytree(file_path, backup_path, dirs_exist_ok=True)

    def delete_path(self, path: Path, backup: bool = False):
        """Delete a file or directory with optional backup."""
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.project_root / self.config["backup_dir"] / timestamp
            self.backup_file(path, backup_dir)

        size = self.get_file_size(path)
        
        if path.is_file():
            path.unlink()
            self.stats["files_deleted"] += 1
        elif path.is_dir():
            shutil.rmtree(path)
            self.stats["dirs_deleted"] += 1
        
        self.stats["space_freed"] += size

    def clean_category(self, category: str, patterns: List[str], 
                      dry_run: bool = False, backup: bool = False,
                      interactive: bool = True) -> Tuple[List[Path], int]:
        """Clean files matching category patterns."""
        files_to_clean = self.find_files(patterns)
        total_size = sum(self.get_file_size(f) for f in files_to_clean)
        
        if not files_to_clean:
            print(f"No {category} files found to clean.")
            return files_to_clean, total_size

        print(f"\n{category.upper()} files found ({len(files_to_clean)} items, {self.format_size(total_size)}):")
        
        for i, file_path in enumerate(files_to_clean[:10]):  # Show first 10
            rel_path = file_path.relative_to(self.project_root)
            size = self.format_size(self.get_file_size(file_path))
            print(f"  {rel_path} ({size})")
        
        if len(files_to_clean) > 10:
            print(f"  ... and {len(files_to_clean) - 10} more files")

        if dry_run:
            print(f"[DRY RUN] Would delete {len(files_to_clean)} {category} files")
            return files_to_clean, total_size

        if interactive:
            response = input(f"Delete these {category} files? [y/N/s(kip)]: ").lower().strip()
            if response == 's':
                return files_to_clean, 0
            elif response != 'y':
                return files_to_clean, 0

        # Delete files
        for file_path in files_to_clean:
            try:
                self.delete_path(file_path, backup)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not delete {file_path}: {e}")

        print(f"✓ Deleted {len(files_to_clean)} {category} files ({self.format_size(total_size)})")
        return files_to_clean, total_size

    def run_cleanup(self, categories: List[str], dry_run: bool = False, 
                   backup: bool = False, interactive: bool = True):
        """Run cleanup for specified categories."""
        print("🧹 CReSO Project Cleanup Tool")
        print(f"Project root: {self.project_root}")
        
        if dry_run:
            print("🔍 DRY RUN MODE - No files will be deleted")
        
        if backup:
            print("💾 Backup mode enabled - files will be backed up before deletion")
        
        total_files_found = 0
        total_size_found = 0
        
        for category in categories:
            if category in self.config["cleanup_patterns"]:
                patterns = self.config["cleanup_patterns"][category]
                files, size = self.clean_category(
                    category, patterns, dry_run, backup, interactive
                )
                total_files_found += len(files)
                total_size_found += size

        print(f"\n📊 Cleanup Summary:")
        print(f"  Files deleted: {self.stats['files_deleted']}")
        print(f"  Directories deleted: {self.stats['dirs_deleted']}")
        print(f"  Space freed: {self.format_size(self.stats['space_freed'])}")
        
        if dry_run:
            print(f"  Total files that would be deleted: {total_files_found}")
            print(f"  Total space that would be freed: {self.format_size(total_size_found)}")

    def list_categories(self):
        """List available cleanup categories."""
        print("Available cleanup categories:")
        for category, patterns in self.config["cleanup_patterns"].items():
            files = self.find_files(patterns)
            total_size = sum(self.get_file_size(f) for f in files)
            print(f"  {category:12} - {len(files):3} items ({self.format_size(total_size)})")


def main():
    parser = argparse.ArgumentParser(description="CReSO Project Cleanup Tool")
    parser.add_argument("categories", nargs="*", 
                       help="Categories to clean (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without deleting")
    parser.add_argument("--backup", action="store_true",
                       help="Backup files before deletion")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Skip confirmation prompts")
    parser.add_argument("--list", action="store_true",
                       help="List available categories and exit")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--save-config", action="store_true",
                       help="Save current configuration and exit")

    args = parser.parse_args()

    cleaner = CReSOCleaner(args.config)

    if args.save_config:
        cleaner.save_config()
        return

    if args.list:
        cleaner.list_categories()
        return

    categories = args.categories or list(cleaner.config["cleanup_patterns"].keys())
    
    # Validate categories
    available_categories = set(cleaner.config["cleanup_patterns"].keys())
    invalid_categories = set(categories) - available_categories
    if invalid_categories:
        print(f"Error: Unknown categories: {', '.join(invalid_categories)}")
        print(f"Available categories: {', '.join(available_categories)}")
        sys.exit(1)

    cleaner.run_cleanup(
        categories=categories,
        dry_run=args.dry_run,
        backup=args.backup,
        interactive=not args.no_interactive
    )


if __name__ == "__main__":
    main()