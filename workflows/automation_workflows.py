import asyncio
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Any
from pathlib import Path

from controllers.email_controller import EmailController
from controllers.file_controller import FileController
from controllers.system_controller import SystemController
from core.automation_engine import AutomationEngine, TaskPriority
from automation.email_automation import EnhancedEmailAutomation

logger = logging.getLogger(__name__)

class AutomationWorkflows:
    """High-level automation workflows combining multiple controllers"""
    
    def __init__(self, llm_client=None):
        self.email_controller = EmailController()
        self.file_controller = FileController()
        self.system_controller = SystemController()
        self.automation_engine = AutomationEngine()
        
        # Initialize enhanced email automation with LLM support
        self.enhanced_email = EnhancedEmailAutomation(llm_client=llm_client)
        
        # Store LLM client for other components
        self.llm_client = llm_client
        
    async def initialize(self):
        """Initialize the automation workflows"""
        await self.automation_engine.start()
        logger.info("Automation workflows initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.automation_engine.stop()
        logger.info("Automation workflows cleaned up")
    
    # Email Automation Workflows
    
    async def send_email_on_behalf(self, to: str, subject: str, body: str,
                                  cc: Optional[str] = None, 
                                  attachments: Optional[List[str]] = None) -> Dict:
        """Send an email on behalf of the user with enhanced features"""
        try:
            # Validate email addresses
            if not self._is_valid_email(to):
                return {"error": f"Invalid email address: {to}"}
            
            if cc and not self._is_valid_email(cc):
                return {"error": f"Invalid CC email address: {cc}"}
            
            # Check attachments exist
            if attachments:
                missing_files = []
                for file_path in attachments:
                    if not Path(file_path).exists():
                        missing_files.append(file_path)
                
                if missing_files:
                    return {"error": f"Attachment files not found: {', '.join(missing_files)}"}
            
            # Send the email
            result = await self.email_controller.send_email(
                to=to, 
                subject=subject, 
                body=body, 
                cc=cc, 
                attachments=attachments
            )
            
            # Log the action
            if result.get("status") == "success":
                logger.info(f"Email sent successfully to {to}: {subject}")
                
                # Optionally send confirmation to user
                await self._send_confirmation_notification(
                    f"Email sent to {to}",
                    f"Your email '{subject}' was sent successfully at {datetime.now().strftime('%I:%M %p')}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in send_email_on_behalf: {e}")
            return {"error": f"Failed to send email: {str(e)}"}
    
    async def send_system_report_email(self, recipient: str, 
                                     include_cleanup_report: bool = False) -> Dict:
        """Send a comprehensive system report via email"""
        try:
            # Get system information
            system_info = await self.system_controller.get_system_info()
            
            if "error" in system_info:
                return system_info
            
            # Create email content
            subject = f"System Report - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
            
            body = self._format_system_report_email(system_info)
            
            # Add cleanup report if requested
            if include_cleanup_report:
                cleanup_result = await self.cleanup_system_files()
                body += "\n\n" + self._format_cleanup_report(cleanup_result)
            
            # Send email
            return await self.send_email_on_behalf(recipient, subject, body)
            
        except Exception as e:
            logger.error(f"Error sending system report email: {e}")
            return {"error": f"Failed to send system report: {str(e)}"}
    
    # System Automation Workflows
    
    async def cleanup_system_files(self, confirm: bool = True) -> Dict:
        """Comprehensive system cleanup workflow"""
        try:
            cleanup_results = {
                "recycle_bin": None,
                "temp_files": None,
                "total_space_freed_mb": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Clear recycle bin
            recycle_result = await self.system_controller.clear_recycle_bin(confirm=confirm)
            cleanup_results["recycle_bin"] = recycle_result
            
            # Clean temp files
            temp_result = await self.system_controller.clean_temp_files(confirm=confirm)
            cleanup_results["temp_files"] = temp_result
            
            # Calculate total space freed
            if temp_result.get("total_size_freed_mb"):
                cleanup_results["total_space_freed_mb"] += temp_result["total_size_freed_mb"]
            
            # Send notification if significant space was freed
            if cleanup_results["total_space_freed_mb"] > 100:  # More than 100MB
                await self._send_confirmation_notification(
                    "System Cleanup Complete",
                    f"Freed {cleanup_results['total_space_freed_mb']:.1f} MB of disk space"
                )
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Error in cleanup_system_files: {e}")
            return {"error": f"System cleanup failed: {str(e)}"}
    
    async def scheduled_system_maintenance(self, 
                                         cleanup_files: bool = True,
                                         send_report: bool = True,
                                         recipient_email: Optional[str] = None) -> Dict:
        """Perform scheduled system maintenance tasks"""
        try:
            maintenance_results = {
                "started_at": datetime.now().isoformat(),
                "tasks_completed": [],
                "tasks_failed": [],
                "summary": {}
            }
            
            # System cleanup
            if cleanup_files:
                try:
                    cleanup_result = await self.cleanup_system_files(confirm=True)
                    maintenance_results["tasks_completed"].append("system_cleanup")
                    maintenance_results["summary"]["cleanup"] = cleanup_result
                except Exception as e:
                    maintenance_results["tasks_failed"].append(f"system_cleanup: {str(e)}")
            
            # Get system info
            try:
                system_info = await self.system_controller.get_system_info()
                maintenance_results["tasks_completed"].append("system_info")
                maintenance_results["summary"]["system_info"] = system_info
            except Exception as e:
                maintenance_results["tasks_failed"].append(f"system_info: {str(e)}")
            
            # Send report email
            if send_report and recipient_email:
                try:
                    email_result = await self.send_system_report_email(
                        recipient_email, 
                        include_cleanup_report=cleanup_files
                    )
                    maintenance_results["tasks_completed"].append("email_report")
                    maintenance_results["summary"]["email_report"] = email_result
                except Exception as e:
                    maintenance_results["tasks_failed"].append(f"email_report: {str(e)}")
            
            maintenance_results["completed_at"] = datetime.now().isoformat()
            maintenance_results["success_rate"] = (
                len(maintenance_results["tasks_completed"]) / 
                (len(maintenance_results["tasks_completed"]) + len(maintenance_results["tasks_failed"]))
                if (maintenance_results["tasks_completed"] or maintenance_results["tasks_failed"]) else 1.0
            )
            
            return maintenance_results
            
        except Exception as e:
            logger.error(f"Error in scheduled_system_maintenance: {e}")
            return {"error": f"Scheduled maintenance failed: {str(e)}"}
    
    # File Management Workflows
    
    async def comprehensive_file_organization(self, scope: str = "common_directories", custom_path: str = None, **kwargs) -> Dict:
        """Comprehensive file organization system"""
        try:
            organization_results = {
                "scope": scope,
                "phase": "analysis",
                "files_analyzed": 0,
                "files_organized": 0,
                "duplicates_found": 0,
                "duplicates_handled": 0,
                "categories_created": [],
                "folders_created": [],
                "errors": [],
                "analysis": {},
                "organization_summary": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Phase 1: Analysis
            logger.info(f"Starting comprehensive file organization for scope: {scope}")
            analysis_result = await self._analyze_file_system(scope, custom_path=custom_path)
            organization_results["analysis"] = analysis_result
            organization_results["files_analyzed"] = analysis_result.get("total_files", 0)
            
            # Phase 2: Planning
            organization_plan = await self._create_organization_plan(analysis_result, **kwargs)
            organization_results["plan"] = organization_plan
            
            # Phase 3: Execution
            organization_results["phase"] = "organization"
            execution_result = await self._execute_organization_plan(organization_plan)
            
            # Update results
            organization_results.update(execution_result)
            organization_results["phase"] = "complete"
            organization_results["success"] = True
            
            # Phase 4: Generate summary report
            summary_report = self._generate_organization_report(organization_results)
            organization_results["final_report"] = summary_report
            
            logger.info(f"File organization complete: {organization_results['files_organized']} files organized")
            return organization_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive file organization: {e}")
            return {
                "error": f"File organization failed: {str(e)}",
                "scope": scope,
                "phase": "error",
                "success": False
            }
    
    async def _analyze_file_system(self, scope: str, custom_path: str = None) -> Dict:
        """Analyze file system to understand current state"""
        analysis = {
            "total_files": 0,
            "total_size": 0,
            "file_types": {},
            "directories_scanned": [],
            "duplicates": [],
            "large_files": [],
            "old_files": [],
            "categories": {
                "documents": [],
                "images": [],
                "videos": [],
                "audio": [],
                "archives": [],
                "executables": [],
                "spreadsheets": [],
                "presentations": [],
                "code": [],
                "other": []
            },
            "recommendations": []
        }
        
        # Determine directories to scan based on scope
        directories_to_scan = self._get_directories_for_scope(scope, custom_path=custom_path)
        
        # Scan each directory
        for directory in directories_to_scan:
            try:
                if Path(directory).exists():
                    # MODIFIED: Only scan files at root level (include_subdirs=False)
                    dir_analysis = await self.file_controller.analyze_directory(
                        directory, include_subdirs=False
                    )
                    
                    if "error" not in dir_analysis:
                        analysis["directories_scanned"].append(directory)
                        analysis["total_files"] += dir_analysis.get("total_files", 0)
                        analysis["total_size"] += dir_analysis.get("total_size", 0)
                        
                        # Merge file types
                        for ext, count in dir_analysis.get("file_types", {}).items():
                            analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + count
                        
                        # Categorize files
                        await self._categorize_files(directory, analysis)
                        
                        # Find duplicates
                        duplicates = await self._find_duplicates(directory)
                        analysis["duplicates"].extend(duplicates)
                        
            except Exception as e:
                logger.warning(f"Error analyzing directory {directory}: {e}")
                analysis["errors"] = analysis.get("errors", [])
                analysis["errors"].append(f"Failed to analyze {directory}: {str(e)}")
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_organization_recommendations(analysis)
        
        return analysis
    
    def _get_directories_for_scope(self, scope: str, custom_path: str = None) -> List[str]:
        """Get list of directories to scan based on scope
        
        RESTRICTED: Only allows Downloads and Documents folders for safety
        """
        
        # Handle custom path first
        if scope == "custom" and custom_path:
            return [custom_path]
        
        home = Path.home()
        
        # RESTRICTED SCOPE: Only Downloads and Documents allowed
        scope_mapping = {
            "Downloads folder": [str(home / "Downloads")],
            "Documents folder": [str(home / "Documents")],
            "common_directories": [
                str(home / "Downloads"),
                str(home / "Documents")
            ],
            "entire system": [
                str(home / "Downloads"),
                str(home / "Documents")
            ],
            # Legacy support - all map to Downloads + Documents only
            "Desktop": [str(home / "Downloads"), str(home / "Documents")],
            "Pictures folder": [str(home / "Downloads"), str(home / "Documents")],
            "Music folder": [str(home / "Downloads"), str(home / "Documents")],
            "Videos folder": [str(home / "Downloads"), str(home / "Documents")],
        }
        
        return scope_mapping.get(scope, scope_mapping["common_directories"])
    
    async def _categorize_files(self, directory: str, analysis: Dict):
        """Categorize files by type and purpose"""
        
        # File type mappings
        category_mappings = {
            "documents": ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.md'],
            "images": ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.tiff'],
            "videos": ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'],
            "audio": ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
            "archives": ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'],
            "executables": ['.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm', '.appimage'],
            "spreadsheets": ['.xls', '.xlsx', '.csv', '.ods'],
            "presentations": ['.ppt', '.pptx', '.odp'],
            "code": ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php', '.rb', '.go']
        }
        
        try:
            # MODIFIED: List files at root level only (recursive=False)
            files_result = await self.file_controller.list_directory(directory, recursive=False)
            
            if "error" not in files_result:
                for file_info in files_result.get("files", []):
                    file_ext = file_info.get("extension", "").lower()
                    
                    # FILTER: Only process files at root level of Downloads/Documents
                    file_path = file_info.get("path", "")
                    base_path = Path(directory)
                    file_path_obj = Path(file_path)
                    
                    # Check if file is directly in the target directory (not in subdirectory)
                    if file_path_obj.parent == base_path:
                        # Find appropriate category
                        categorized = False
                        for category, extensions in category_mappings.items():
                            if file_ext in extensions:
                                analysis["categories"][category].append(file_info)
                                categorized = True
                                break
                        
                        # If not categorized, add to other
                        if not categorized:
                            analysis["categories"]["other"].append(file_info)
                        
        except Exception as e:
            logger.warning(f"Error categorizing files in {directory}: {e}")
    
    async def _find_duplicates(self, directory: str) -> List[Dict]:
        """Find duplicate files in directory"""
        duplicates = []
        file_hashes = {}
        
        try:
            # MODIFIED: Find duplicates only at root level (recursive=False)
            files_result = await self.file_controller.list_directory(directory, recursive=False)
            
            if "error" not in files_result:
                for file_info in files_result.get("files", []):
                    file_path = file_info.get("path")
                    file_size = file_info.get("size", 0)
                    
                    # FILTER: Only process files at root level of Downloads/Documents
                    base_path = Path(directory)
                    file_path_obj = Path(file_path)
                    
                    # Check if file is directly in the target directory (not in subdirectory)
                    if file_path_obj.parent != base_path:
                        continue
                    
                    # Skip very small files and system files
                    if file_size < 1024 or file_info.get("name", "").startswith("."):
                        continue
                    
                    try:
                        # Calculate hash for duplicate detection
                        hash_result = await self.file_controller.get_file_hash(file_path)
                        
                        if "error" not in hash_result:
                            file_hash = hash_result.get("hash")
                            
                            if file_hash in file_hashes:
                                # Found duplicate
                                duplicates.append({
                                    "original": file_hashes[file_hash],
                                    "duplicate": file_info,
                                    "hash": file_hash,
                                    "size": file_size
                                })
                            else:
                                file_hashes[file_hash] = file_info
                                
                    except Exception as e:
                        logger.warning(f"Error calculating hash for {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Error finding duplicates in {directory}: {e}")
        
        return duplicates
    
    def _generate_organization_recommendations(self, analysis: Dict) -> List[str]:
        """Generate organization recommendations based on analysis"""
        recommendations = []
        
        total_files = analysis.get("total_files", 0)
        duplicates_count = len(analysis.get("duplicates", []))
        
        if total_files > 1000:
            recommendations.append("Large number of files detected - organization will significantly improve system performance")
        
        if duplicates_count > 0:
            recommendations.append(f"Found {duplicates_count} duplicate files that can be cleaned up to save space")
        
        # Check for scattered file types
        categories = analysis.get("categories", {})
        if len(categories.get("other", [])) > total_files * 0.3:
            recommendations.append("Many uncategorized files found - custom organization rules may be beneficial")
        
        if len(categories.get("documents", [])) > 100:
            recommendations.append("Large number of documents found - consider organizing by date or project")
        
        if len(categories.get("images", [])) > 200:
            recommendations.append("Large image collection detected - consider organizing by date or event")
        
        return recommendations
    
    async def _create_organization_plan(self, analysis: Dict, **kwargs) -> Dict:
        """Create organization plan based on analysis - IN-PLACE organization only"""
        plan = {
            "create_categories": kwargs.get("create_categories", True),
            "organize_duplicates": kwargs.get("organize_duplicates", True),
            "preserve_structure": kwargs.get("preserve_structure", False),
            "backup_before_organize": kwargs.get("backup_before_organize", False),
            "category_structure": {},
            "duplicate_actions": [],
            "file_moves": [],
            "folders_to_create": [],
            "in_place_organization": True  # NEW: Flag for in-place organization
        }
        
        if plan["create_categories"]:
            # MODIFIED: Plan category folder structure WITHIN existing directories
            directories_scanned = analysis.get("directories_scanned", [])
            
            for directory in directories_scanned:
                base_path = Path(directory)
                
                # Only organize within Downloads and Documents folders (or custom path for testing)
                if not (any(folder in str(base_path).lower() for folder in ['downloads', 'documents']) or 
                       str(base_path).startswith(tempfile.gettempdir())):
                    continue
                
                # Create category subfolders within the existing directory
                for category, files in analysis.get("categories", {}).items():
                    if files and category != "other":
                        # Filter files that belong to this directory AND are at root level
                        directory_files = [
                            f for f in files 
                            if f.get("path", "").startswith(str(base_path)) and 
                            Path(f.get("path", "")).parent == base_path  # Root level only
                        ]
                        
                        if directory_files:
                            category_path = base_path / f"Organized_{category.title()}"
                            plan["category_structure"][f"{directory}_{category}"] = str(category_path)
                            plan["folders_to_create"].append(str(category_path))
                            
                            # Plan file moves within the same directory
                            for file_info in directory_files:
                                source_path = file_info.get("path")
                                file_name = Path(source_path).name
                                dest_path = category_path / file_name
                                
                                # Only move if it's not already in a category folder
                                if not any(cat in str(source_path) for cat in ['Organized_Documents', 'Organized_Images', 'Organized_Videos', 'Organized_Audio', 'Organized_Archives']):
                                    plan["file_moves"].append({
                                        "source": source_path,
                                        "destination": str(dest_path),
                                        "category": category,
                                        "size": file_info.get("size", 0),
                                        "directory": str(base_path)
                                    })
        
        if plan["organize_duplicates"]:
            # MODIFIED: Handle duplicates within the same directories
            directories_scanned = analysis.get("directories_scanned", [])
            
            for directory in directories_scanned:
                base_path = Path(directory)
                
                # Only handle duplicates within Downloads and Documents (or custom path for testing)
                if not (any(folder in str(base_path).lower() for folder in ['downloads', 'documents']) or 
                       str(base_path).startswith(tempfile.gettempdir())):
                    continue
                
                duplicate_path = base_path / "Duplicates"
                plan["folders_to_create"].append(str(duplicate_path))
                
                # Plan duplicate handling within this directory
                for duplicate in analysis.get("duplicates", []):
                    duplicate_file_path = duplicate["duplicate"]["path"]
                    
                    # Only handle duplicates from this directory AND at root level
                    if (duplicate_file_path.startswith(str(base_path)) and 
                        Path(duplicate_file_path).parent == base_path):
                        plan["duplicate_actions"].append({
                            "action": "move_to_duplicates",
                            "original": duplicate["original"]["path"],
                            "duplicate": duplicate_file_path,
                            "destination": str(duplicate_path / duplicate["duplicate"]["name"]),
                            "size_saved": duplicate["size"],
                            "directory": str(base_path)
                        })
        
        return plan
    
    async def _execute_organization_plan(self, plan: Dict) -> Dict:
        """Execute the organization plan"""
        execution_results = {
            "files_organized": 0,
            "folders_created": 0,
            "duplicates_handled": 0,
            "space_saved": 0,
            "errors": [],
            "categories_created": [],
            "execution_log": []
        }
        
        try:
            # Create folder structure
            for folder_path in plan.get("folders_to_create", []):
                try:
                    Path(folder_path).mkdir(parents=True, exist_ok=True)
                    execution_results["folders_created"] += 1
                    execution_results["execution_log"].append(f"Created folder: {folder_path}")
                    
                    # Track categories created
                    category_name = Path(folder_path).name
                    if category_name not in execution_results["categories_created"]:
                        execution_results["categories_created"].append(category_name)
                        
                except Exception as e:
                    error_msg = f"Failed to create folder {folder_path}: {str(e)}"
                    execution_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Execute file moves
            for move_operation in plan.get("file_moves", []):
                try:
                    source = move_operation["source"]
                    destination = move_operation["destination"]
                    
                    # Ensure destination directory exists
                    Path(destination).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move file
                    move_result = await self.file_controller.move_file(source, destination)
                    
                    if "error" not in move_result:
                        execution_results["files_organized"] += 1
                        execution_results["execution_log"].append(
                            f"Moved {Path(source).name} to {move_operation['category']}"
                        )
                    else:
                        execution_results["errors"].append(
                            f"Failed to move {source}: {move_result['error']}"
                        )
                        
                except Exception as e:
                    error_msg = f"Error moving file {move_operation['source']}: {str(e)}"
                    execution_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Handle duplicates
            for duplicate_action in plan.get("duplicate_actions", []):
                try:
                    if duplicate_action["action"] == "move_to_duplicates":
                        source = duplicate_action["duplicate"]
                        destination = duplicate_action["destination"]
                        
                        # Move duplicate to duplicates folder
                        move_result = await self.file_controller.move_file(source, destination)
                        
                        if "error" not in move_result:
                            execution_results["duplicates_handled"] += 1
                            execution_results["space_saved"] += duplicate_action["size_saved"]
                            execution_results["execution_log"].append(
                                f"Moved duplicate: {Path(source).name}"
                            )
                        else:
                            execution_results["errors"].append(
                                f"Failed to move duplicate {source}: {move_result['error']}"
                            )
                            
                except Exception as e:
                    error_msg = f"Error handling duplicate {duplicate_action['duplicate']}: {str(e)}"
                    execution_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Error executing organization plan: {e}")
            execution_results["errors"].append(f"Execution failed: {str(e)}")
            return execution_results
    
    def _generate_organization_report(self, results: Dict) -> str:
        """Generate a comprehensive organization report for in-place organization"""
        report_lines = [
            "📁 **File Organization Complete! (Root Level Only)**",
            "",
            "ℹ️  **Scope:** Only files directly in Downloads and Documents folders (not subfolders)",
            "",
            "📊 **Summary:**"
        ]
        
        # Add statistics
        files_analyzed = results.get("files_analyzed", 0)
        files_organized = results.get("files_organized", 0)
        folders_created = results.get("folders_created", 0)
        duplicates_handled = results.get("duplicates_handled", 0)
        space_saved = results.get("space_saved", 0)
        
        report_lines.extend([
            f"• {files_analyzed} files analyzed",
            f"• {files_organized} files organized within existing folders",
            f"• {folders_created} category subfolders created",
            f"• {duplicates_handled} duplicates handled",
            f"• {space_saved // (1024*1024)} MB space saved" if space_saved > 0 else "• No space saved"
        ])
        
        # Add organization locations
        directories_organized = set()
        execution_log = results.get("execution_log", [])
        for log_entry in execution_log:
            if "Created folder:" in log_entry:
                folder_path = log_entry.replace("Created folder: ", "")
                parent_dir = str(Path(folder_path).parent)
                if any(folder in parent_dir.lower() for folder in ['downloads', 'documents']):
                    directories_organized.add(parent_dir)
        
        if directories_organized:
            report_lines.extend([
                "",
                "📂 **Organized Within:**"
            ])
            for directory in sorted(directories_organized):
                dir_name = Path(directory).name
                report_lines.append(f"• {dir_name} folder")
        
        # Add categories created
        categories = results.get("categories_created", [])
        if categories:
            report_lines.extend([
                "",
                "📁 **Category Subfolders Created:**"
            ])
            for category in categories:
                if category.startswith("Organized_"):
                    clean_name = category.replace("Organized_", "")
                    report_lines.append(f"• {clean_name}/")
                else:
                    report_lines.append(f"• {category}/")
        
        # Add errors if any
        errors = results.get("errors", [])
        if errors:
            report_lines.extend([
                "",
                "⚠️ **Issues Encountered:**"
            ])
            for error in errors[:5]:  # Show first 5 errors
                report_lines.append(f"• {error}")
            
            if len(errors) > 5:
                report_lines.append(f"• ... and {len(errors) - 5} more issues")
        
        # Add note about in-place organization
        report_lines.extend([
            "",
            "ℹ️ **Note:** Files were organized within their original folders (Downloads/Documents only)"
        ])
        
        return "\n".join(report_lines)
    
    async def organize_downloads_folder(self, downloads_path: Optional[str] = None) -> Dict:
        """Organize files in downloads folder by type"""
        try:
            if not downloads_path:
                downloads_path = str(Path.home() / "Downloads")
            
            if not Path(downloads_path).exists():
                return {"error": f"Downloads folder not found: {downloads_path}"}
            
            # Define organization structure
            file_categories = {
                "Documents": ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
                "Images": ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
                "Videos": ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
                "Audio": ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
                "Archives": ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
                "Executables": ['.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm'],
                "Spreadsheets": ['.xls', '.xlsx', '.csv', '.ods'],
                "Presentations": ['.ppt', '.pptx', '.odp']
            }
            
            organization_results = {
                "downloads_path": downloads_path,
                "files_organized": 0,
                "folders_created": [],
                "categories": {},
                "errors": []
            }
            
            # List files in downloads
            files_result = await self.file_controller.list_directory(downloads_path)
            
            if "error" in files_result:
                return files_result
            
            # Create category folders and move files
            for file_info in files_result.get("files", []):
                file_path = file_info["path"]
                file_ext = file_info["extension"].lower()
                
                # Find appropriate category
                target_category = "Other"
                for category, extensions in file_categories.items():
                    if file_ext in extensions:
                        target_category = category
                        break
                
                # Create category folder if needed
                category_path = Path(downloads_path) / target_category
                if not category_path.exists():
                    category_path.mkdir()
                    organization_results["folders_created"].append(target_category)
                
                # Move file
                try:
                    new_path = str(category_path / file_info["name"])
                    move_result = await self.file_controller.move_file(file_path, new_path)
                    
                    if "error" not in move_result:
                        organization_results["files_organized"] += 1
                        
                        if target_category not in organization_results["categories"]:
                            organization_results["categories"][target_category] = 0
                        organization_results["categories"][target_category] += 1
                    else:
                        organization_results["errors"].append(f"Failed to move {file_info['name']}: {move_result['error']}")
                        
                except Exception as e:
                    organization_results["errors"].append(f"Error moving {file_info['name']}: {str(e)}")
            
            organization_results["timestamp"] = datetime.now().isoformat()
            
            return organization_results
            
        except Exception as e:
            logger.error(f"Error organizing downloads folder: {e}")
            return {"error": f"Failed to organize downloads: {str(e)}"}
    
    # Utility Methods
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    async def _send_confirmation_notification(self, subject: str, message: str):
        """Send a confirmation notification to the user"""
        try:
            await self.email_controller.send_notification_email(subject, message)
        except Exception as e:
            logger.warning(f"Failed to send confirmation notification: {e}")
    
    def _format_system_report_email(self, system_info: Dict) -> str:
        """Format system information for email"""
        if "error" in system_info:
            return f"Error getting system information: {system_info['error']}"
        
        report = f"""System Report - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}

SYSTEM INFORMATION:
OS: {system_info['system']['os']} {system_info['system']['os_version']}
Architecture: {system_info['system']['architecture']}
Hostname: {system_info['system']['hostname']}

HARDWARE:
CPU: {system_info['hardware']['cpu_count']} cores, {system_info['hardware']['cpu_percent']}% usage
Memory: {system_info['hardware']['memory']['available_gb']:.1f}GB available of {system_info['hardware']['memory']['total_gb']:.1f}GB total ({system_info['hardware']['memory']['percent_used']}% used)

STORAGE:"""
        
        for storage in system_info.get('storage', []):
            report += f"""
{storage['device']} ({storage['filesystem']}): {storage['free_gb']:.1f}GB free of {storage['total_gb']:.1f}GB ({storage['percent_used']}% used)"""
        
        report += f"""

PROCESSES:
Total running processes: {system_info['processes']['total']}

TOP CPU PROCESSES:"""
        
        for proc in system_info['processes']['top_cpu']:
            report += f"""
{proc['name']} (PID: {proc['pid']}) - {proc['cpu_percent']}% CPU"""
        
        report += f"""

Report generated at: {system_info['timestamp']}
"""
        
        return report
    
    def _format_cleanup_report(self, cleanup_result: Dict) -> str:
        """Format cleanup results for email"""
        if "error" in cleanup_result:
            return f"Cleanup Error: {cleanup_result['error']}"
        
        report = """
SYSTEM CLEANUP REPORT:
"""
        
        # Recycle bin cleanup
        recycle_result = cleanup_result.get("recycle_bin", {})
        if recycle_result.get("status") == "success":
            report += f"✓ Recycle bin cleared successfully\n"
        else:
            report += f"✗ Recycle bin cleanup failed: {recycle_result.get('error', 'Unknown error')}\n"
        
        # Temp files cleanup
        temp_result = cleanup_result.get("temp_files", {})
        if temp_result.get("status") == "success":
            report += f"✓ Temporary files cleaned: {temp_result.get('total_size_freed_mb', 0):.1f}MB freed\n"
            
            locations = temp_result.get("locations_cleaned", [])
            if locations:
                report += "  Cleaned locations:\n"
                for location in locations:
                    report += f"    - {location['path']}: {location['size_freed_mb']:.1f}MB\n"
        else:
            report += f"✗ Temp files cleanup failed: {temp_result.get('error', 'Unknown error')}\n"
        
        report += f"\nTotal space freed: {cleanup_result.get('total_space_freed_mb', 0):.1f}MB\n"
        report += f"Cleanup completed at: {cleanup_result.get('timestamp', 'Unknown')}\n"
        
        return report
    
    # Quick Action Methods
    
    async def quick_email(self, to: str, subject: str, body: str) -> Dict:
        """Quick email sending with minimal parameters"""
        return await self.send_email_on_behalf(to, subject, body)
    
    async def quick_cleanup(self) -> Dict:
        """Quick system cleanup with default settings"""
        return await self.cleanup_system_files(confirm=True)
    
    async def quick_system_info(self) -> Dict:
        """Quick system information retrieval"""
        return await self.system_controller.get_system_info()
    
    # Scheduled Tasks (for future implementation with task scheduler)
    
    async def setup_daily_maintenance(self, time_str: str = "02:00", 
                                    email_recipient: Optional[str] = None) -> Dict:
        """Setup daily maintenance schedule (placeholder)"""
        return {
            "status": "scheduled",
            "message": f"Daily maintenance scheduled for {time_str}",
            "tasks": ["system_cleanup", "system_report"],
            "email_recipient": email_recipient,
            "note": "Actual scheduling system not yet implemented"
        }
    
    async def setup_weekly_report(self, day: str = "Sunday", 
                                time_str: str = "09:00",
                                email_recipient: Optional[str] = None) -> Dict:
        """Setup weekly system report (placeholder)"""
        return {
            "status": "scheduled",
            "message": f"Weekly report scheduled for {day} at {time_str}",
            "email_recipient": email_recipient,
            "note": "Actual scheduling system not yet implemented"
        }
    
    # Enhanced Email Automation Methods

    
    async def process_intelligent_email_request(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """Process email request using enhanced email automation with improved content generation"""
        try:
            if hasattr(self, 'enhanced_email') and self.enhanced_email:
                logger.info("Using Enhanced Email Automation for intelligent processing")
                
                # Configure user name if provided in context
                if context and context.get("user_name"):
                    self.enhanced_email.set_user_name(context["user_name"])
                
                # Auto-send when called from web interface (no terminal interaction available)
                auto_send = context.get("auto_send", True) if context else True  # Default to True for web interface
                
                logger.info(f"Processing email with auto_send={auto_send}")
                
                # Process the request through enhanced automation
                result = await self.enhanced_email.process_email_request(user_input, context or {}, auto_send=auto_send)
                
                # Ensure proper final_output for consistent display
                if "final_output" not in result and result.get("status") == "success":
                    result["final_output"] = result.get("message", "Email processed successfully")
                
                return result
            else:
                logger.warning("Enhanced email automation not available, falling back to basic parsing")
                # Fallback to basic email processing
                email_info = self._parse_basic_email_info(user_input)
                
                if not email_info.get("to"):
                    return {
                        "status": "error",
                        "error": "Could not identify email recipient. Please specify an email address.",
                        "final_output": "❌ Could not identify email recipient. Please specify an email address.",
                        "suggestion": "Try: 'Send email to user@example.com about [topic]'"
                    }
                
                return await self.send_email_on_behalf(
                    to=email_info["to"],
                    subject=email_info.get("subject", "Message from AI Assistant"),
                    body=email_info.get("body", "This message was sent via AI Assistant.")
                )
                
        except Exception as e:
            logger.error(f"Error in intelligent email processing: {e}")
            return {
                "status": "error", 
                "error": f"Email processing failed: {str(e)}",
                "final_output": f"❌ Email processing failed: {str(e)}"
            }
    
    def _parse_basic_email_info(self, user_input: str) -> Dict[str, str]:
        """Parse basic email information from user input as fallback"""
        import re
        
        email_info = {}
        
        # Extract email address (to)
        email_pattern = r'(?:to|send to|email to|mail to)\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        email_match = re.search(email_pattern, user_input, re.IGNORECASE)
        if email_match:
            email_info["to"] = email_match.group(1)
        else:
            # Try to find any email address in the input
            email_pattern_generic = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            email_matches = re.findall(email_pattern_generic, user_input)
            if email_matches:
                email_info["to"] = email_matches[0]
        
        # Extract subject (explicit)
        subject_pattern = r'(?:subject|with subject)\s+["\']([^"\']]+)["\']'
        subject_match = re.search(subject_pattern, user_input, re.IGNORECASE)
        if subject_match:
            email_info["subject"] = subject_match.group(1)
        else:
            # Try without quotes
            subject_pattern2 = r'(?:subject|with subject)\s+([^,\.]+)'
            subject_match2 = re.search(subject_pattern2, user_input, re.IGNORECASE)
            if subject_match2:
                email_info["subject"] = subject_match2.group(1).strip()
        
        # Extract body content after "telling" or similar phrases
        body_patterns = [
            r'telling (?:him|her|them) (?:that )?(.+)',
            r'saying (?:that )?(.+)',
            r'informing (?:him|her|them) (?:that )?(.+)',
            r'letting (?:him|her|them) know (?:that )?(.+)',
            r'with (?:the )?message (.+)',
            r'about (.+)',
            r'regarding (.+)'
        ]
        
        for pattern in body_patterns:
            body_match = re.search(pattern, user_input, re.IGNORECASE)
            if body_match:
                email_info["body"] = body_match.group(1).strip()
                break
        
        # Generate subject from body if not explicitly provided
        if "body" in email_info and "subject" not in email_info:
            body_text = email_info["body"]
            
            # Generate subject based on content
            if "meeting" in body_text.lower():
                if "can't" in body_text.lower() or "won't" in body_text.lower() or "unable" in body_text.lower():
                    email_info["subject"] = "Unable to Attend Meeting"
                else:
                    email_info["subject"] = "Regarding Meeting"
            elif "emergency" in body_text.lower():
                email_info["subject"] = "Important: Emergency Situation"
            elif "update" in body_text.lower():
                email_info["subject"] = "Update"
            else:
                # Use first few words as subject
                words = body_text.split()[:6]
                email_info["subject"] = " ".join(words)
                if len(body_text.split()) > 6:
                    email_info["subject"] += "..."
        
        # Default subject if still none
        if "subject" not in email_info:
            email_info["subject"] = "Message from AI Assistant"
        
        # Clean up the body text
        if "body" in email_info:
            body = email_info["body"]
            
            # Add greeting if not present
            if not any(greeting in body.lower() for greeting in ["hi", "hello", "dear"]):
                body = "Hi,\n\n" + body
            
            # Add closing if not present
            if not any(closing in body.lower() for closing in ["regards", "thanks", "sincerely", "best"]):
                body += "\n\nBest regards"
            
            email_info["body"] = body
        else:
            # Default body
            email_info["body"] = "Hi,\n\nI hope this message finds you well.\n\nBest regards"
        
        return email_info
    
    def _parse_basic_email_info(self, user_input: str) -> Dict[str, str]:
        """Parse basic email information from user input"""
        import re
        
        email_info = {}
        
        # Extract email address
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        email_match = re.search(email_pattern, user_input)
        if email_match:
            email_info["to"] = email_match.group(1)
        
        # Extract subject from common patterns
        subject_patterns = [
            r'subject[:\s]+([^,\n]+)',
            r'about\s+([^,\n]+)',
            r'regarding\s+([^,\n]+)'
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                email_info["subject"] = match.group(1).strip()
                break
        
        # Use entire input as body if no specific subject found
        if "subject" not in email_info:
            email_info["body"] = user_input
        
        return email_info
    
    # Legacy Email Automation Workflows (Enhanced)
    
    async def send_email_with_llm_content(self, to: str = None, subject: str = None, 
                                        body: str = None, original_request: str = "", 
                                        llm_analysis: Dict = None) -> Dict:
        """Send email using LLM-generated content through enhanced automation"""
        try:
            # Use enhanced email automation if available
            if hasattr(self, 'enhanced_email') and self.enhanced_email:
                # Create a formatted request for enhanced processing
                if original_request:
                    request = original_request
                else:
                    request = f"Send email to {to or '[recipient]'} with subject '{subject or '[subject]'}': {body or '[content]'}"
                
                return await self.enhanced_email.process_email_request(request, auto_send=True)
            else:
                # Fallback to email controller method
                return await self.email_controller.send_email_with_llm_content(
                    to=to, subject=subject, body=body, 
                    original_request=original_request, llm_analysis=llm_analysis
                )
        except Exception as e:
            logger.error(f"Error in LLM-guided email workflow: {e}")
            return {"error": f"Email workflow failed: {str(e)}"}
    
    async def execute_system_control_with_llm(self, operation: str = None, volume: int = None, 
                                            application: str = None, llm_analysis: Dict = None) -> Dict:
        """Execute system control using LLM analysis through system controller"""
        try:
            return await self.system_controller.execute_system_control_with_llm(
                operation=operation, volume=volume, application=application, llm_analysis=llm_analysis
            )
        except Exception as e:
            logger.error(f"Error in LLM-guided system control workflow: {e}")
            return {"error": f"System control workflow failed: {str(e)}"}
    
    async def process_llm_guided_request(self, llm_analysis: Dict) -> Dict:
        """Process any request using LLM analysis to determine the appropriate workflow"""
        
        try:
            action_type = llm_analysis.get('action_type', 'automation_workflow')
            parameters = llm_analysis.get('parameters', {})
            
            print(f"\n🤖 **Processing LLM-Guided Request**")
            print(f"**Action Type:** {action_type}")
            print(f"**Intent:** {llm_analysis.get('intent', 'Unknown')}")
            print(f"**Confidence:** {llm_analysis.get('confidence', 0):.0%}")
            
            if action_type == 'email':
                return await self.send_email_with_llm_content(
                    to=parameters.get('to'),
                    subject=parameters.get('subject'),
                    body=parameters.get('body'),
                    original_request=parameters.get('original_request', ''),
                    llm_analysis=llm_analysis
                )
            
            elif action_type == 'system_control':
                return await self.execute_system_control_with_llm(
                    operation=parameters.get('operation'),
                    volume=parameters.get('volume'),
                    application=parameters.get('application'),
                    llm_analysis=llm_analysis
                )
            
            elif action_type == 'file_operation':
                # For now, return a placeholder for file operations
                return {
                    "status": "not_implemented",
                    "message": "File operations with LLM guidance not yet implemented",
                    "llm_analysis": llm_analysis,
                    "suggestion": "Please use specific file management commands"
                }
            
            elif action_type == 'automation_workflow':
                # General automation workflow
                description = parameters.get('description', '')
                
                return {
                    "status": "processed",
                    "message": f"General automation request processed: {description}",
                    "llm_analysis": llm_analysis,
                    "suggestion": "For better results, please be more specific about the action you want to perform"
                }
            
            else:
                return {
                    "status": "unknown_action",
                    "message": f"Unknown action type: {action_type}",
                    "llm_analysis": llm_analysis,
                    "available_actions": ["email", "system_control", "file_operation", "automation_workflow"]
                }
                
        except Exception as e:
            logger.error(f"Error processing LLM-guided request: {e}")
            return {
                "error": f"Failed to process LLM-guided request: {str(e)}",
                "llm_analysis": llm_analysis
            }