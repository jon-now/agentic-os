import os
import shutil
import hashlib
from pathlib import Path
import logging
from datetime import datetime, timedelta
import mimetypes
import zipfile
import tarfile
from typing import Dict, List
import json

logger = logging.getLogger(__name__)

class FileController:
    def __init__(self):
        self.allowed_operations = {
            'read', 'write', 'copy', 'move', 'delete', 'create_dir',
            'list_dir', 'search', 'compress', 'extract', 'analyze'
        }
        self.safe_extensions = {
            '.txt', '.md', '.json', '.csv', '.xml', '.yaml', '.yml',
            '.log', '.cfg', '.con', '.ini', '.py', '.js', '.html',
            '.css', '.sql', '.sh', '.bat', '.ps1'
        }

    async def list_directory(self, path: str, recursive: bool = False,
                           show_hidden: bool = False, max_depth: int = 3) -> Dict:
        """List directory contents"""
        try:
            path_obj = Path(path).resolve()

            if not path_obj.exists():
                return {"error": f"Path does not exist: {path}"}

            if not path_obj.is_dir():
                return {"error": f"Path is not a directory: {path}"}

            files = []
            directories = []

            if recursive:
                items = self._list_recursive(path_obj, max_depth, show_hidden)
            else:
                items = list(path_obj.iterdir())

            for item in items:
                if not show_hidden and item.name.startswith('.'):
                    continue

                try:
                    stat = item.stat()
                    item_info = {
                        "name": item.name,
                        "path": str(item),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "is_directory": item.is_dir(),
                        "extension": item.suffix.lower() if not item.is_dir() else "",
                        "mime_type": mimetypes.guess_type(str(item))[0] if not item.is_dir() else None
                    }

                    if item.is_dir():
                        directories.append(item_info)
                    else:
                        files.append(item_info)

                except (PermissionError, OSError) as e:
                    logger.warning("Cannot access {item}: %s", e)
                    continue

            return {
                "path": str(path_obj),
                "files": files,
                "directories": directories,
                "total_files": len(files),
                "total_directories": len(directories),
                "total_size": sum(f["size"] for f in files),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error listing directory: %s", e)
            return {"error": f"Directory listing failed: {str(e)}"}

    async def read_file(self, file_path: str, encoding: str = 'utf-8',
                       max_size_mb: int = 10) -> Dict:
        """Read file contents"""
        try:
            path_obj = Path(file_path).resolve()

            if not path_obj.exists():
                return {"error": f"File does not exist: {file_path}"}

            if not path_obj.is_file():
                return {"error": f"Path is not a file: {file_path}"}

            # Check file size
            file_size = path_obj.stat().st_size
            if file_size > max_size_mb * 1024 * 1024:
                return {"error": f"File too large: {file_size / (1024*1024):.1f}MB > {max_size_mb}MB"}

            # Check if file extension is safe to read
            if path_obj.suffix.lower() not in self.safe_extensions:
                return {"error": f"File type not supported for reading: {path_obj.suffix}"}

            # Read file
            try:
                content = path_obj.read_text(encoding=encoding)
            except UnicodeDecodeError:
                # Try binary read for non-text files
                content = path_obj.read_bytes().hex()
                encoding = 'binary'

            return {
                "file_path": str(path_obj),
                "content": content,
                "size": file_size,
                "encoding": encoding,
                "lines": len(content.split('\n')) if encoding != 'binary' else 0,
                "extension": path_obj.suffix.lower(),
                "mime_type": mimetypes.guess_type(str(path_obj))[0],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error reading file: %s", e)
            return {"error": f"File reading failed: {str(e)}"}

    async def write_file(self, file_path: str, content: str,
                        encoding: str = 'utf-8', backup: bool = True) -> Dict:
        """Write content to file"""
        try:
            path_obj = Path(file_path).resolve()

            # Create backup if file exists and backup is requested
            if backup and path_obj.exists():
                backup_path = path_obj.with_suffix(path_obj.suffix + '.backup')
                shutil.copy2(path_obj, backup_path)

            # Create parent directories if they don't exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            path_obj.write_text(content, encoding=encoding)

            return {
                "file_path": str(path_obj),
                "size": len(content.encode(encoding)),
                "lines": len(content.split('\n')),
                "encoding": encoding,
                "backup_created": backup and path_obj.exists(),
                "status": "written",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error writing file: %s", e)
            return {"error": f"File writing failed: {str(e)}"}

    async def copy_file(self, source_path: str, destination_path: str,
                       overwrite: bool = False) -> Dict:
        """Copy file or directory"""
        try:
            source_obj = Path(source_path).resolve()
            dest_obj = Path(destination_path).resolve()

            if not source_obj.exists():
                return {"error": f"Source does not exist: {source_path}"}

            if dest_obj.exists() and not overwrite:
                return {"error": f"Destination exists and overwrite=False: {destination_path}"}

            # Create parent directories
            dest_obj.parent.mkdir(parents=True, exist_ok=True)

            if source_obj.is_file():
                shutil.copy2(source_obj, dest_obj)
                operation = "file_copied"
            else:
                shutil.copytree(source_obj, dest_obj, dirs_exist_ok=overwrite)
                operation = "directory_copied"

            return {
                "source": str(source_obj),
                "destination": str(dest_obj),
                "operation": operation,
                "size": self._get_size(dest_obj),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error copying: %s", e)
            return {"error": f"Copy operation failed: {str(e)}"}

    async def move_file(self, source_path: str, destination_path: str,
                       overwrite: bool = False) -> Dict:
        """Move file or directory"""
        try:
            source_obj = Path(source_path).resolve()
            dest_obj = Path(destination_path).resolve()

            if not source_obj.exists():
                return {"error": f"Source does not exist: {source_path}"}

            if dest_obj.exists() and not overwrite:
                return {"error": f"Destination exists and overwrite=False: {destination_path}"}

            # Create parent directories
            dest_obj.parent.mkdir(parents=True, exist_ok=True)

            # Move file/directory
            shutil.move(str(source_obj), str(dest_obj))

            return {
                "source": str(source_obj),
                "destination": str(dest_obj),
                "operation": "moved",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error moving: %s", e)
            return {"error": f"Move operation failed: {str(e)}"}

    async def delete_file(self, file_path: str, permanent: bool = False) -> Dict:
        """Delete file or directory"""
        try:
            path_obj = Path(file_path).resolve()

            if not path_obj.exists():
                return {"error": f"Path does not exist: {file_path}"}

            # Safety check - don't delete system directories
            system_paths = {'/bin', '/usr', '/etc', '/sys', '/proc', 'C:\\Windows', 'C:\\Program Files'}
            if any(str(path_obj).startswith(sys_path) for sys_path in system_paths):
                return {"error": "Cannot delete system directories"}

            size = self._get_size(path_obj)
            is_directory = path_obj.is_dir()

            if not permanent:
                # Move to trash/recycle bin (simplified implementation)
                trash_dir = Path.home() / '.trash'
                trash_dir.mkdir(exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                trash_path = trash_dir / f"{path_obj.name}_{timestamp}"

                shutil.move(str(path_obj), str(trash_path))
                operation = "moved_to_trash"
                final_path = str(trash_path)
            else:
                if is_directory:
                    shutil.rmtree(path_obj)
                else:
                    path_obj.unlink()
                operation = "permanently_deleted"
                final_path = None

            return {
                "original_path": str(path_obj),
                "final_path": final_path,
                "operation": operation,
                "was_directory": is_directory,
                "size_deleted": size,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error deleting: %s", e)
            return {"error": f"Delete operation failed: {str(e)}"}

    async def search_files(self, search_path: str, pattern: str,
                          search_content: bool = False, max_results: int = 100) -> Dict:
        """Search for files by name or content"""
        try:
            path_obj = Path(search_path).resolve()

            if not path_obj.exists() or not path_obj.is_dir():
                return {"error": f"Search path is not a valid directory: {search_path}"}

            results = []

            # Search by filename
            for file_path in path_obj.rglob(pattern):
                if len(results) >= max_results:
                    break

                try:
                    stat = file_path.stat()
                    result = {
                        "path": str(file_path),
                        "name": file_path.name,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_directory": file_path.is_dir(),
                        "match_type": "filename"
                    }

                    # Search content if requested and file is text
                    if (search_content and file_path.is_file() and
                        file_path.suffix.lower() in self.safe_extensions):
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            if pattern.lower() in content.lower():
                                result["match_type"] = "content"
                                # Find line numbers with matches
                                lines = content.split('\n')
                                matching_lines = [
                                    (i+1, line.strip()) for i, line in enumerate(lines)
                                    if pattern.lower() in line.lower()
                                ]
                                result["matching_lines"] = matching_lines[:5]  # First 5 matches
                        except Exception:
                            pass  # Skip files that can't be read

                    results.append(result)

                except (PermissionError, OSError):
                    continue

            return {
                "search_path": str(path_obj),
                "pattern": pattern,
                "search_content": search_content,
                "results": results,
                "total_found": len(results),
                "max_results_reached": len(results) >= max_results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error searching files: %s", e)
            return {"error": f"File search failed: {str(e)}"}

    async def analyze_directory(self, path: str, include_subdirs: bool = True) -> Dict:
        """Analyze directory structure and file types"""
        try:
            path_obj = Path(path).resolve()

            if not path_obj.exists() or not path_obj.is_dir():
                return {"error": f"Path is not a valid directory: {path}"}

            analysis = {
                "path": str(path_obj),
                "total_files": 0,
                "total_directories": 0,
                "total_size": 0,
                "file_types": {},
                "size_distribution": {
                    "small": 0,    # < 1MB
                    "medium": 0,   # 1MB - 100MB
                    "large": 0     # > 100MB
                },
                "recent_files": [],
                "largest_files": [],
                "oldest_files": [],
                "newest_files": []
            }

            all_files = []

            # Walk through directory
            for item in path_obj.rglob('*') if include_subdirs else path_obj.iterdir():
                try:
                    stat = item.stat()

                    if item.is_file():
                        analysis["total_files"] += 1
                        analysis["total_size"] += stat.st_size

                        # File type analysis
                        ext = item.suffix.lower() or 'no_extension'
                        analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1

                        # Size distribution
                        size_mb = stat.st_size / (1024 * 1024)
                        if size_mb < 1:
                            analysis["size_distribution"]["small"] += 1
                        elif size_mb < 100:
                            analysis["size_distribution"]["medium"] += 1
                        else:
                            analysis["size_distribution"]["large"] += 1

                        # Collect file info for further analysis
                        file_info = {
                            "path": str(item),
                            "name": item.name,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                        all_files.append(file_info)

                    elif item.is_dir():
                        analysis["total_directories"] += 1

                except (PermissionError, OSError):
                    continue

            # Sort files for top lists
            all_files.sort(key=lambda x: x["size"], reverse=True)
            analysis["largest_files"] = all_files[:10]

            all_files.sort(key=lambda x: x["modified"])
            analysis["oldest_files"] = all_files[:5]
            analysis["newest_files"] = all_files[-5:]

            # Recent files (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            analysis["recent_files"] = [
                f for f in all_files
                if datetime.fromtimestamp(f["modified"]) > week_ago
            ][:20]

            # Summary statistics
            analysis["average_file_size"] = (
                analysis["total_size"] / analysis["total_files"]
                if analysis["total_files"] > 0 else 0
            )

            analysis["timestamp"] = datetime.now().isoformat()

            return analysis

        except Exception as e:
            logger.error("Error analyzing directory: %s", e)
            return {"error": f"Directory analysis failed: {str(e)}"}

    async def compress_files(self, file_paths: List[str], output_path: str,
                           compression_type: str = 'zip') -> Dict:
        """Compress files into archive"""
        try:
            output_obj = Path(output_path).resolve()

            # Validate input files
            valid_paths = []
            for file_path in file_paths:
                path_obj = Path(file_path).resolve()
                if path_obj.exists():
                    valid_paths.append(path_obj)

            if not valid_paths:
                return {"error": "No valid files to compress"}

            # Create archive
            if compression_type.lower() == 'zip':
                with zipfile.ZipFile(output_obj, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for path_obj in valid_paths:
                        if path_obj.is_file():
                            zipf.write(path_obj, path_obj.name)
                        else:
                            for file_path in path_obj.rglob('*'):
                                if file_path.is_file():
                                    arcname = file_path.relative_to(path_obj.parent)
                                    zipf.write(file_path, arcname)

            elif compression_type.lower() in ['tar', 'tar.gz', 'tgz']:
                mode = 'w:gz' if compression_type.lower() in ['tar.gz', 'tgz'] else 'w'
                with tarfile.open(output_obj, mode) as tarf:
                    for path_obj in valid_paths:
                        tarf.add(path_obj, path_obj.name)

            else:
                return {"error": f"Unsupported compression type: {compression_type}"}

            return {
                "archive_path": str(output_obj),
                "compression_type": compression_type,
                "files_compressed": len(valid_paths),
                "archive_size": output_obj.stat().st_size,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error compressing files: %s", e)
            return {"error": f"Compression failed: {str(e)}"}

    async def extract_archive(self, archive_path: str, extract_to: str) -> Dict:
        """Extract archive contents"""
        try:
            archive_obj = Path(archive_path).resolve()
            extract_obj = Path(extract_to).resolve()

            if not archive_obj.exists():
                return {"error": f"Archive does not exist: {archive_path}"}

            # Create extraction directory
            extract_obj.mkdir(parents=True, exist_ok=True)

            extracted_files = []

            # Extract based on file type
            if archive_obj.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_obj, 'r') as zipf:
                    zipf.extractall(extract_obj)
                    extracted_files = zipf.namelist()

            elif archive_obj.suffix.lower() in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_obj, 'r:*') as tarf:
                    tarf.extractall(extract_obj)
                    extracted_files = tarf.getnames()

            else:
                return {"error": f"Unsupported archive type: {archive_obj.suffix}"}

            return {
                "archive_path": str(archive_obj),
                "extract_path": str(extract_obj),
                "files_extracted": len(extracted_files),
                "extracted_files": extracted_files[:20],  # First 20 files
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error extracting archive: %s", e)
            return {"error": f"Extraction failed: {str(e)}"}

    async def get_file_hash(self, file_path: str, algorithm: str = 'sha256') -> Dict:
        """Calculate file hash"""
        try:
            path_obj = Path(file_path).resolve()

            if not path_obj.exists() or not path_obj.is_file():
                return {"error": f"File does not exist: {file_path}"}

            # Calculate hash
            hash_obj = hashlib.new(algorithm)

            with open(path_obj, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)

            return {
                "file_path": str(path_obj),
                "algorithm": algorithm,
                "hash": hash_obj.hexdigest(),
                "file_size": path_obj.stat().st_size,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error calculating hash: %s", e)
            return {"error": f"Hash calculation failed: {str(e)}"}

    def _list_recursive(self, path: Path, max_depth: int, show_hidden: bool, current_depth: int = 0):
        """Recursively list directory contents with depth limit"""
        items = []

        if current_depth >= max_depth:
            return items

        try:
            for item in path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue

                items.append(item)

                if item.is_dir() and current_depth < max_depth - 1:
                    items.extend(self._list_recursive(item, max_depth, show_hidden, current_depth + 1))
        except PermissionError:
            pass

        return items

    def _get_size(self, path: Path) -> int:
        """Get total size of file or directory"""
        if path.is_file():
            return path.stat().st_size

        total_size = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except (PermissionError, OSError):
            pass

        return total_size
