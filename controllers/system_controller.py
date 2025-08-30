import os
import shutil
import subprocess
import platform
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import psutil
import platform

if platform.system() == "Windows":
    import winreg
else:
    winreg = None

logger = logging.getLogger(__name__)

class SystemController:
    """Controller for system-level automation tasks"""
    
    def __init__(self):
        self.system = platform.system()
        self.is_windows = self.system == "Windows"
        self.is_linux = self.system == "Linux"
        self.is_mac = self.system == "Darwin"
        
    async def clear_recycle_bin(self, confirm: bool = False) -> Dict:
        """Clear the system recycle bin/trash"""
        try:
            if not confirm:
                return {
                    "error": "Confirmation required to clear recycle bin",
                    "message": "Set confirm=True to proceed with clearing recycle bin"
                }
            
            if self.is_windows:
                return await self._clear_windows_recycle_bin()
            elif self.is_mac:
                return await self._clear_mac_trash()
            elif self.is_linux:
                return await self._clear_linux_trash()
            else:
                return {"error": f"Unsupported operating system: {self.system}"}
                
        except Exception as e:
            logger.error(f"Error clearing recycle bin: {e}")
            return {"error": f"Failed to clear recycle bin: {str(e)}"}
    
    async def _clear_windows_recycle_bin(self) -> Dict:
        """Clear Windows recycle bin with improved error handling"""
        try:
            print("🗑️ Attempting to clear Windows Recycle Bin...")
            
            # Try multiple methods in order of preference
            methods_to_try = [
                self._clear_windows_recycle_bin_powershell,
                self._clear_windows_recycle_bin_vbscript,
                self._clear_windows_recycle_bin_alternative
            ]
            
            for i, method in enumerate(methods_to_try, 1):
                try:
                    print(f"   Trying method {i}/{len(methods_to_try)}...")
                    result = await method()
                    
                    if result.get("status") == "success":
                        print(f"   ✅ Success with method {i}")
                        return result
                    else:
                        print(f"   ⚠️ Method {i} failed: {result.get('error', 'Unknown error')}")
                        continue
                        
                except Exception as e:
                    print(f"   ❌ Method {i} exception: {str(e)}")
                    continue
            
            # If all methods failed
            return {
                "status": "error",
                "error": "All recycle bin clearing methods failed",
                "message": "Unable to clear recycle bin. It may be empty or access is restricted.",
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Error in recycle bin clearing: {e}")
            return {"error": f"Failed to clear recycle bin: {str(e)}"}
    
    async def _clear_windows_recycle_bin_powershell(self) -> Dict:
        """Clear Windows recycle bin using PowerShell with smart pre-check"""
        try:
            print("      Running PowerShell Clear-RecycleBin...")
            
            # First, check if recycle bin has any items
            check_cmd = [
                "powershell", "-ExecutionPolicy", "Bypass",
                "-Command", "(New-Object -ComObject Shell.Application).Namespace(10).Items().Count"
            ]
            
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            
            if check_result.returncode == 0:
                item_count = int(check_result.stdout.strip() or "0")
                print(f"        Recycle bin contains {item_count} items")
                
                if item_count == 0:
                    return {
                        "status": "success",
                        "message": "Windows Recycle Bin is already empty",
                        "method": "PowerShell (pre-check)",
                        "items_removed": 0,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Use PowerShell to clear recycle bin with reduced timeout
            cmd = [
                "powershell", "-ExecutionPolicy", "Bypass", 
                "-Command", "Clear-RecycleBin -Force -Confirm:$false"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": "Windows Recycle Bin cleared successfully",
                    "method": "PowerShell Clear-RecycleBin",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # If Clear-RecycleBin fails but we know there are items, try COM method
                if 'item_count' in locals() and item_count > 0:
                    print("        Clear-RecycleBin failed, trying COM method...")
                    return await self._clear_windows_recycle_bin_powershell_com()
                
                error_msg = result.stderr.strip() if result.stderr else "Unknown PowerShell error"
                return {
                    "status": "error",
                    "error": f"PowerShell failed: {error_msg}"
                }
                
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "PowerShell command timed out"}
        except Exception as e:
            return {"status": "error", "error": f"PowerShell method failed: {str(e)}"}
    
    async def _clear_windows_recycle_bin_powershell_com(self) -> Dict:
        """PowerShell method using COM objects (more reliable)"""
        try:
            com_script = '''
            try {
                $shell = New-Object -ComObject Shell.Application
                $recycleBin = $shell.Namespace(10)
                $items = $recycleBin.Items()
                
                if ($items.Count -gt 0) {
                    foreach ($item in $items) {
                        $item.InvokeVerb("delete")
                    }
                    Write-Host "SUCCESS: Cleared $($items.Count) items"
                } else {
                    Write-Host "SUCCESS: Recycle bin is already empty"
                }
            } catch {
                Write-Host "ERROR: $($_.Exception.Message)"
                exit 1
            }
            '''
            
            cmd = [
                "powershell", "-ExecutionPolicy", "Bypass",
                "-Command", com_script
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": "Windows Recycle Bin cleared successfully",
                    "method": "PowerShell COM",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "error": f"PowerShell COM failed: {result.stderr.strip() if result.stderr else 'Unknown error'}"
                }
                
        except Exception as e:
            return {"status": "error", "error": f"PowerShell COM method failed: {str(e)}"}
    
    async def _clear_windows_recycle_bin_vbscript(self) -> Dict:
        """Clear Windows recycle bin using VBScript (alternative method)"""
        try:
            # Create a temporary VBScript to empty recycle bin
            vbscript_code = '''Set objShell = CreateObject("Shell.Application")
Set objFolder = objShell.Namespace(10)
For Each objFolderItem In objFolder.Items
    objFolderItem.InvokeVerb "delete"
Next'''
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.vbs', delete=False) as f:
                f.write(vbscript_code)
                vbs_path = f.name
            
            try:
                print("      Running VBScript method...")
                cmd = ["cscript", "//NoLogo", vbs_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                
                if result.returncode == 0:
                    return {
                        "status": "success",
                        "message": "Windows Recycle Bin cleared successfully",
                        "method": "VBScript",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"VBScript failed: {result.stderr.strip() if result.stderr else 'Unknown error'}"
                    }
            finally:
                try:
                    os.unlink(vbs_path)
                except:
                    pass
                
        except Exception as e:
            return {"status": "error", "error": f"VBScript method failed: {str(e)}"}
    
    async def _clear_windows_recycle_bin_alternative(self) -> Dict:
        """Alternative method to clear Windows recycle bin using direct file operations"""
        try:
            print("      Using direct file system method...")
            
            # Get all available drives
            drives = []
            for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
                drive = f"{letter}:"
                if os.path.exists(drive):
                    drives.append(drive)
            
            cleared_drives = []
            total_items_removed = 0
            
            for drive in drives:
                recycle_path = f"{drive}\\$Recycle.Bin"
                if os.path.exists(recycle_path):
                    try:
                        print(f"        Checking {drive} recycle bin...")
                        
                        # Count items before deletion
                        items_count = 0
                        for root, dirs, files in os.walk(recycle_path):
                            items_count += len(files)
                        
                        if items_count > 0:
                            print(f"        Found {items_count} items in {drive} recycle bin")
                            
                            # Try to remove individual files first
                            removed_count = 0
                            for root, dirs, files in os.walk(recycle_path, topdown=False):
                                for file in files:
                                    try:
                                        file_path = os.path.join(root, file)
                                        os.remove(file_path)
                                        removed_count += 1
                                    except (PermissionError, FileNotFoundError):
                                        continue
                                
                                # Remove empty directories
                                for dir_name in dirs:
                                    try:
                                        dir_path = os.path.join(root, dir_name)
                                        if not os.listdir(dir_path):
                                            os.rmdir(dir_path)
                                    except (PermissionError, OSError):
                                        continue
                            
                            if removed_count > 0:
                                print(f"        Removed {removed_count} items from {drive}")
                                cleared_drives.append(drive)
                                total_items_removed += removed_count
                        else:
                            print(f"        {drive} recycle bin is already empty")
                            
                    except Exception as e:
                        print(f"        Failed to process {drive}: {str(e)}")
                        continue
            
            if cleared_drives or total_items_removed > 0:
                return {
                    "status": "success",
                    "message": f"Recycle bin cleared for drives: {', '.join(cleared_drives)} ({total_items_removed} items removed)",
                    "method": "Direct file system operations",
                    "drives_cleared": cleared_drives,
                    "items_removed": total_items_removed,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "success",
                    "message": "Recycle bin is already empty or no accessible items found",
                    "method": "Direct file system operations",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {"status": "error", "error": f"Alternative method failed: {str(e)}"}
    
    async def _clear_mac_trash(self) -> Dict:
        """Clear macOS trash"""
        try:
            trash_path = Path.home() / ".Trash"
            
            if not trash_path.exists():
                return {
                    "status": "success",
                    "message": "Trash is already empty",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Count items before deletion
            items_count = len(list(trash_path.iterdir()))
            
            # Clear trash
            shutil.rmtree(trash_path)
            trash_path.mkdir()
            
            return {
                "status": "success",
                "message": f"macOS Trash cleared successfully ({items_count} items removed)",
                "items_removed": items_count,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to clear macOS trash: {str(e)}"}
    
    async def _clear_linux_trash(self) -> Dict:
        """Clear Linux trash"""
        try:
            trash_path = Path.home() / ".local/share/Trash"
            
            if not trash_path.exists():
                return {
                    "status": "success",
                    "message": "Trash is already empty",
                    "timestamp": datetime.now().isoformat()
                }
            
            items_removed = 0
            
            # Clear files and info directories
            for subdir in ["files", "info"]:
                subdir_path = trash_path / subdir
                if subdir_path.exists():
                    items_count = len(list(subdir_path.iterdir()))
                    shutil.rmtree(subdir_path)
                    subdir_path.mkdir()
                    items_removed += items_count
            
            return {
                "status": "success",
                "message": f"Linux Trash cleared successfully ({items_removed} items removed)",
                "items_removed": items_removed,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to clear Linux trash: {str(e)}"}
    
    async def clean_temp_files(self, confirm: bool = False) -> Dict:
        """Clean temporary files from the system"""
        try:
            if not confirm:
                return {
                    "error": "Confirmation required to clean temp files",
                    "message": "Set confirm=True to proceed with cleaning temporary files"
                }
            
            cleaned_locations = []
            total_size_freed = 0
            
            if self.is_windows:
                temp_paths = [
                    os.environ.get('TEMP', ''),
                    os.environ.get('TMP', ''),
                    'C:\\Windows\\Temp',
                    f"{os.environ.get('USERPROFILE', '')}\\AppData\\Local\\Temp"
                ]
            else:
                temp_paths = ['/tmp', '/var/tmp', f"{Path.home()}/tmp"]
            
            for temp_path in temp_paths:
                if temp_path and os.path.exists(temp_path):
                    try:
                        size_freed = await self._clean_directory(temp_path)
                        if size_freed > 0:
                            cleaned_locations.append({
                                "path": temp_path,
                                "size_freed_mb": round(size_freed / (1024 * 1024), 2)
                            })
                            total_size_freed += size_freed
                    except Exception as e:
                        logger.warning(f"Could not clean {temp_path}: {e}")
                        continue
            
            return {
                "status": "success",
                "message": f"Temporary files cleaned successfully",
                "locations_cleaned": cleaned_locations,
                "total_size_freed_mb": round(total_size_freed / (1024 * 1024), 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
            return {"error": f"Failed to clean temp files: {str(e)}"}
    
    async def _clean_directory(self, directory_path: str) -> int:
        """Clean files from a directory and return bytes freed"""
        total_size = 0
        
        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        total_size += file_size
                    except (PermissionError, FileNotFoundError, OSError):
                        continue
                        
                # Remove empty directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except (PermissionError, OSError):
                        continue
                        
        except Exception as e:
            logger.warning(f"Error cleaning directory {directory_path}: {e}")
            
        return total_size
    
    async def shutdown_system(self, delay_minutes: int = 0, confirm: bool = False) -> Dict:
        """Shutdown the system with optional delay"""
        try:
            if not confirm:
                return {
                    "error": "Confirmation required for system shutdown",
                    "message": "Set confirm=True to proceed with system shutdown"
                }
            
            if self.is_windows:
                cmd = ["shutdown", "/s", "/t", str(delay_minutes * 60)]
            elif self.is_linux or self.is_mac:
                if delay_minutes > 0:
                    cmd = ["sudo", "shutdown", "-h", f"+{delay_minutes}"]
                else:
                    cmd = ["sudo", "shutdown", "-h", "now"]
            else:
                return {"error": f"Shutdown not supported on {self.system}"}
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"System shutdown scheduled in {delay_minutes} minutes" if delay_minutes > 0 else "System shutting down now",
                    "delay_minutes": delay_minutes,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"Shutdown command failed: {result.stderr}"}
                
        except Exception as e:
            logger.error(f"Error scheduling shutdown: {e}")
            return {"error": f"Failed to schedule shutdown: {str(e)}"}
    
    async def restart_system(self, delay_minutes: int = 0, confirm: bool = False) -> Dict:
        """Restart the system with optional delay"""
        try:
            if not confirm:
                return {
                    "error": "Confirmation required for system restart",
                    "message": "Set confirm=True to proceed with system restart"
                }
            
            if self.is_windows:
                cmd = ["shutdown", "/r", "/t", str(delay_minutes * 60)]
            elif self.is_linux or self.is_mac:
                if delay_minutes > 0:
                    cmd = ["sudo", "shutdown", "-r", f"+{delay_minutes}"]
                else:
                    cmd = ["sudo", "shutdown", "-r", "now"]
            else:
                return {"error": f"Restart not supported on {self.system}"}
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"System restart scheduled in {delay_minutes} minutes" if delay_minutes > 0 else "System restarting now",
                    "delay_minutes": delay_minutes,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"Restart command failed: {result.stderr}"}
                
        except Exception as e:
            logger.error(f"Error scheduling restart: {e}")
            return {"error": f"Failed to schedule restart: {str(e)}"}
    
    async def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        try:
            # Basic system info
            info = {
                "system": {
                    "os": platform.system(),
                    "os_version": platform.version(),
                    "architecture": platform.architecture()[0],
                    "processor": platform.processor(),
                    "hostname": platform.node(),
                    "python_version": platform.python_version()
                },
                "hardware": {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory": {
                        "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                        "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                        "percent_used": psutil.virtual_memory().percent
                    }
                },
                "storage": [],
                "network": {
                    "interfaces": len(psutil.net_if_addrs()),
                    "connections": len(psutil.net_connections())
                },
                "processes": {
                    "total": len(psutil.pids()),
                    "top_cpu": []
                }
            }
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    info["storage"].append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "filesystem": partition.fstype,
                        "total_gb": round(usage.total / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                        "percent_used": round((usage.used / usage.total) * 100, 1)
                    })
                except PermissionError:
                    continue
            
            # Top CPU processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            info["processes"]["top_cpu"] = processes[:5]
            
            info["timestamp"] = datetime.now().isoformat()
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": f"Failed to get system info: {str(e)}"}
    
    async def kill_process(self, process_name: str = None, pid: int = None, confirm: bool = False) -> Dict:
        """Kill a process by name or PID"""
        try:
            if not confirm:
                return {
                    "error": "Confirmation required to kill process",
                    "message": "Set confirm=True to proceed with killing process"
                }
            
            if not process_name and not pid:
                return {"error": "Either process_name or pid must be provided"}
            
            killed_processes = []
            
            if pid:
                try:
                    proc = psutil.Process(pid)
                    proc_name = proc.name()
                    proc.terminate()
                    killed_processes.append({"pid": pid, "name": proc_name})
                except psutil.NoSuchProcess:
                    return {"error": f"Process with PID {pid} not found"}
                except psutil.AccessDenied:
                    return {"error": f"Access denied to kill process {pid}"}
            
            elif process_name:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if process_name.lower() in proc.info['name'].lower():
                            proc.terminate()
                            killed_processes.append({
                                "pid": proc.info['pid'],
                                "name": proc.info['name']
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            if killed_processes:
                return {
                    "status": "success",
                    "message": f"Killed {len(killed_processes)} process(es)",
                    "killed_processes": killed_processes,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "no_action",
                    "message": f"No processes found matching criteria",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error killing process: {e}")
            return {"error": f"Failed to kill process: {str(e)}"}
    
    async def set_volume(self, level: int) -> Dict:
        """Set system volume level (0-100)"""
        try:
            if not 0 <= level <= 100:
                return {"error": "Volume level must be between 0 and 100"}
            
            if self.is_windows:
                # Use PowerShell with proper volume control for Windows
                try:
                    # Method 1: Use PowerShell with Windows Audio API
                    powershell_script = f"""
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class VolumeControl {{
    [DllImport("user32.dll")]
    public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
    
    public static void SetVolume(int level) {{
        // Mute first
        keybd_event(0xAD, 0, 0, UIntPtr.Zero);
        keybd_event(0xAD, 0, 2, UIntPtr.Zero);
        
        // Set to 0
        for(int i = 0; i < 50; i++) {{
            keybd_event(0xAE, 0, 0, UIntPtr.Zero);
            keybd_event(0xAE, 0, 2, UIntPtr.Zero);
        }}
        
        // Set to desired level
        int steps = level / 2;
        for(int i = 0; i < steps; i++) {{
            keybd_event(0xAF, 0, 0, UIntPtr.Zero);
            keybd_event(0xAF, 0, 2, UIntPtr.Zero);
        }}
    }}
}}
'@
[VolumeControl]::SetVolume({level})
"""
                    
                    cmd = ["powershell", "-Command", powershell_script]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        return {
                            "status": "success",
                            "message": f"Volume set to {level}%",
                            "level": level,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        # Fallback method using nircmd if available
                        return await self._set_volume_nircmd(level)
                        
                except subprocess.TimeoutExpired:
                    return {"error": "Volume control timed out"}
                except Exception as e:
                    logger.warning(f"PowerShell volume control failed: {e}")
                    return await self._set_volume_nircmd(level)
            
            elif self.is_linux:
                # Use amixer for Linux
                try:
                    cmd = ["amixer", "set", "Master", f"{level}%"]
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    
                    return {
                        "status": "success",
                        "message": f"Volume set to {level}%",
                        "level": level,
                        "timestamp": datetime.now().isoformat()
                    }
                except subprocess.CalledProcessError:
                    return {"error": "amixer not available or failed"}
            
            elif self.is_mac:
                # Use osascript for macOS
                try:
                    cmd = ["osascript", "-e", f"set volume output volume {level}"]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    return {
                        "status": "success",
                        "message": f"Volume set to {level}%",
                        "level": level,
                        "timestamp": datetime.now().isoformat()
                    }
                except subprocess.CalledProcessError:
                    return {"error": "Volume control failed on macOS"}
            
            else:
                return {"error": f"Volume control not supported on {self.system}"}
                
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return {"error": f"Failed to set volume: {str(e)}"}
    
    async def _set_volume_nircmd(self, level: int) -> Dict:
        """Fallback volume control using nircmd"""
        try:
            # Try to use nircmd if available
            cmd = ["nircmd.exe", "setsysvolume", str(int(level * 655.35))]  # Convert to 0-65535 range
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Volume set to {level}% (via nircmd)",
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "partial",
                    "message": f"Volume control attempted but may not be accurate. Install nircmd for better control.",
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }
                
        except subprocess.TimeoutExpired:
            return {"error": "nircmd volume control timed out"}
        except FileNotFoundError:
            return {
                "status": "partial", 
                "message": f"Volume control attempted. For better control, install nircmd.exe",
                "level": level,
                "timestamp": datetime.now().isoformat()
            }
    
    async def lock_screen(self) -> Dict:
        """Lock the system screen"""
        try:
            if self.is_windows:
                cmd = ["rundll32.exe", "user32.dll,LockWorkStation"]
            elif self.is_mac:
                cmd = ["pmset", "displaysleepnow"]
            elif self.is_linux:
                # Try common Linux screen lockers
                for locker in ["gnome-screensaver-command", "xscreensaver-command", "i3lock"]:
                    if shutil.which(locker):
                        if locker == "gnome-screensaver-command":
                            cmd = [locker, "--lock"]
                        elif locker == "xscreensaver-command":
                            cmd = [locker, "-lock"]
                        else:
                            cmd = [locker]
                        break
                else:
                    return {"error": "No screen locker found on this Linux system"}
            else:
                return {"error": f"Screen lock not supported on {self.system}"}
            
            subprocess.run(cmd, check=True)
            
            return {
                "status": "success",
                "message": "Screen locked successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error locking screen: {e}")
            return {"error": f"Failed to lock screen: {str(e)}"}
    
    async def open_application(self, app_name: str) -> Dict:
        """Open an application by name"""
        try:
            if not app_name or not app_name.strip():
                return {"error": "Application name cannot be empty"}
            
            app_name = app_name.strip()
            
            if self.is_windows:
                return await self._open_windows_application(app_name)
            elif self.is_mac:
                return await self._open_mac_application(app_name)
            elif self.is_linux:
                return await self._open_linux_application(app_name)
            else:
                return {"error": f"Application launching not supported on {self.system}"}
                
        except Exception as e:
            logger.error(f"Error opening application: {e}")
            return {"error": f"Failed to open application: {str(e)}"}
    
    async def _open_windows_application(self, app_name: str) -> Dict:
        """Open application on Windows"""
        try:
            app_name_lower = app_name.lower()
            
            # Common application mappings for Windows
            app_mappings = {
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "paint": "mspaint.exe",
                "cmd": "cmd.exe",
                "command prompt": "cmd.exe",
                "powershell": "powershell.exe",
                "task manager": "taskmgr.exe",
                "control panel": "control.exe",
                "registry editor": "regedit.exe",
                "regedit": "regedit.exe",
                "file explorer": "explorer.exe",
                "explorer": "explorer.exe",
                "chrome": "chrome.exe",
                "firefox": "firefox.exe",
                "edge": "msedge.exe",
                "microsoft edge": "msedge.exe",
                "word": "winword.exe",
                "excel": "excel.exe",
                "powerpoint": "powerpnt.exe",
                "outlook": "outlook.exe",
                "teams": "teams.exe",
                "skype": "skype.exe",
                "discord": "discord.exe",
                "spotify": "spotify.exe",
                "steam": "steam.exe",
                "vlc": "vlc.exe",
                "photoshop": "photoshop.exe",
                "vs code": "code.exe",
                "visual studio code": "code.exe",
                "vscode": "code.exe",
                "sublime": "sublime_text.exe",
                "atom": "atom.exe",
                "git bash": "git-bash.exe",
                "putty": "putty.exe",
                "wireshark": "wireshark.exe",
                "7zip": "7zFM.exe",
                "winrar": "winrar.exe"
            }
            
            # Try direct mapping first
            executable = app_mappings.get(app_name_lower, app_name)
            
            # Method 1: Try to run directly
            try:
                result = subprocess.run([executable], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or "is not recognized" not in result.stderr:
                    return {
                        "status": "success",
                        "message": f"Opened {app_name}",
                        "executable": executable,
                        "method": "direct",
                        "timestamp": datetime.now().isoformat()
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Method 2: Try with start command
            try:
                cmd = ["start", "", executable]
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return {
                        "status": "success",
                        "message": f"Opened {app_name}",
                        "executable": executable,
                        "method": "start_command",
                        "timestamp": datetime.now().isoformat()
                    }
            except subprocess.TimeoutExpired:
                pass
            
            # Method 3: Try PowerShell Start-Process
            try:
                cmd = ["powershell", "-Command", f"Start-Process '{executable}'"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return {
                        "status": "success",
                        "message": f"Opened {app_name}",
                        "executable": executable,
                        "method": "powershell",
                        "timestamp": datetime.now().isoformat()
                    }
            except subprocess.TimeoutExpired:
                pass
            
            # Method 4: Search in common directories
            search_paths = [
                "C:\\Program Files",
                "C:\\Program Files (x86)",
                f"C:\\Users\\{os.environ.get('USERNAME', '')}\\AppData\\Local",
                f"C:\\Users\\{os.environ.get('USERNAME', '')}\\AppData\\Roaming"
            ]
            
            found_executable = await self._find_executable_windows(app_name, search_paths)
            if found_executable:
                try:
                    subprocess.Popen([found_executable])
                    return {
                        "status": "success",
                        "message": f"Opened {app_name}",
                        "executable": found_executable,
                        "method": "search_and_launch",
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Failed to launch found executable {found_executable}: {e}")
            
            # Method 5: Try Windows Run dialog approach
            try:
                cmd = ["powershell", "-Command", f"Start-Process -FilePath '{app_name}'"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return {
                        "status": "success",
                        "message": f"Opened {app_name}",
                        "executable": app_name,
                        "method": "powershell_filepath",
                        "timestamp": datetime.now().isoformat()
                    }
            except subprocess.TimeoutExpired:
                pass
            
            return {
                "error": f"Could not find or launch application: {app_name}",
                "suggestions": list(app_mappings.keys())[:10]
            }
            
        except Exception as e:
            logger.error(f"Error opening Windows application: {e}")
            return {"error": f"Failed to open Windows application: {str(e)}"}
    
    async def _find_executable_windows(self, app_name: str, search_paths: List[str]) -> Optional[str]:
        """Find executable in Windows directories"""
        try:
            app_name_lower = app_name.lower()
            
            for search_path in search_paths:
                if not os.path.exists(search_path):
                    continue
                
                try:
                    for root, dirs, files in os.walk(search_path):
                        # Limit search depth to avoid taking too long
                        level = root.replace(search_path, '').count(os.sep)
                        if level >= 3:
                            dirs[:] = []  # Don't go deeper
                            continue
                        
                        for file in files:
                            if file.lower().endswith('.exe'):
                                file_lower = file.lower()
                                # Check if app name is in the executable name
                                if (app_name_lower in file_lower or 
                                    file_lower.startswith(app_name_lower) or
                                    app_name_lower.replace(' ', '') in file_lower.replace(' ', '')):
                                    return os.path.join(root, file)
                                    
                except (PermissionError, OSError):
                    continue
                    
        except Exception as e:
            logger.warning(f"Error searching for executable: {e}")
            
        return None
    
    async def _open_mac_application(self, app_name: str) -> Dict:
        """Open application on macOS"""
        try:
            # Try using 'open' command
            cmd = ["open", "-a", app_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Opened {app_name}",
                    "method": "open_command",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"Could not open {app_name}: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            return {"error": f"Timeout opening {app_name}"}
        except Exception as e:
            return {"error": f"Failed to open macOS application: {str(e)}"}
    
    async def _open_linux_application(self, app_name: str) -> Dict:
        """Open application on Linux"""
        try:
            # Try different methods for Linux
            methods = [
                [app_name],  # Direct command
                ["which", app_name],  # Check if command exists
                ["xdg-open", app_name],  # XDG open
                ["gnome-open", app_name],  # GNOME
                ["kde-open", app_name],  # KDE
            ]
            
            # First check if command exists
            try:
                result = subprocess.run(["which", app_name], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Command exists, try to run it
                    subprocess.Popen([app_name])
                    return {
                        "status": "success",
                        "message": f"Opened {app_name}",
                        "method": "direct_command",
                        "timestamp": datetime.now().isoformat()
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Try desktop file approach
            try:
                cmd = ["gtk-launch", app_name]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return {
                        "status": "success",
                        "message": f"Opened {app_name}",
                        "method": "gtk_launch",
                        "timestamp": datetime.now().isoformat()
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            return {"error": f"Could not find or launch application: {app_name}"}
            
        except Exception as e:
            return {"error": f"Failed to open Linux application: {str(e)}"}
    
    async def list_installed_applications(self) -> Dict:
        """List installed applications"""
        try:
            if self.is_windows:
                return await self._list_windows_applications()
            elif self.is_mac:
                return await self._list_mac_applications()
            elif self.is_linux:
                return await self._list_linux_applications()
            else:
                return {"error": f"Application listing not supported on {self.system}"}
                
        except Exception as e:
            logger.error(f"Error listing applications: {e}")
            return {"error": f"Failed to list applications: {str(e)}"}
    
    async def _list_windows_applications(self) -> Dict:
        """List Windows applications"""
        try:
            applications = []
            
            # Method 1: Check Start Menu
            start_menu_paths = [
                "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs",
                f"C:\\Users\\{os.environ.get('USERNAME', '')}\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs"
            ]
            
            for start_path in start_menu_paths:
                if os.path.exists(start_path):
                    try:
                        for root, dirs, files in os.walk(start_path):
                            for file in files:
                                if file.endswith('.lnk'):
                                    app_name = file[:-4]  # Remove .lnk extension
                                    applications.append({
                                        "name": app_name,
                                        "path": os.path.join(root, file),
                                        "type": "shortcut"
                                    })
                    except (PermissionError, OSError):
                        continue
            
            # Method 2: Check Program Files
            program_paths = ["C:\\Program Files", "C:\\Program Files (x86)"]
            
            for prog_path in program_paths:
                if os.path.exists(prog_path):
                    try:
                        for item in os.listdir(prog_path):
                            item_path = os.path.join(prog_path, item)
                            if os.path.isdir(item_path):
                                applications.append({
                                    "name": item,
                                    "path": item_path,
                                    "type": "program_folder"
                                })
                    except (PermissionError, OSError):
                        continue
            
            # Remove duplicates and sort
            unique_apps = {}
            for app in applications:
                name = app["name"].lower()
                if name not in unique_apps:
                    unique_apps[name] = app
            
            sorted_apps = sorted(unique_apps.values(), key=lambda x: x["name"].lower())
            
            return {
                "applications": sorted_apps[:50],  # Limit to first 50
                "total_found": len(sorted_apps),
                "system": "Windows",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error listing Windows applications: {e}")
            return {"error": f"Failed to list Windows applications: {str(e)}"}
    
    async def _list_mac_applications(self) -> Dict:
        """List macOS applications"""
        try:
            applications = []
            app_dirs = ["/Applications", f"{Path.home()}/Applications"]
            
            for app_dir in app_dirs:
                if os.path.exists(app_dir):
                    try:
                        for item in os.listdir(app_dir):
                            if item.endswith('.app'):
                                app_name = item[:-4]  # Remove .app extension
                                applications.append({
                                    "name": app_name,
                                    "path": os.path.join(app_dir, item),
                                    "type": "application"
                                })
                    except (PermissionError, OSError):
                        continue
            
            sorted_apps = sorted(applications, key=lambda x: x["name"].lower())
            
            return {
                "applications": sorted_apps,
                "total_found": len(sorted_apps),
                "system": "macOS",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to list macOS applications: {str(e)}"}
    
    async def _list_linux_applications(self) -> Dict:
        """List Linux applications"""
        try:
            applications = []
            
            # Check desktop files
            desktop_dirs = [
                "/usr/share/applications",
                f"{Path.home()}/.local/share/applications"
            ]
            
            for desktop_dir in desktop_dirs:
                if os.path.exists(desktop_dir):
                    try:
                        for file in os.listdir(desktop_dir):
                            if file.endswith('.desktop'):
                                app_name = file[:-8]  # Remove .desktop extension
                                applications.append({
                                    "name": app_name,
                                    "path": os.path.join(desktop_dir, file),
                                    "type": "desktop_file"
                                })
                    except (PermissionError, OSError):
                        continue
            
            sorted_apps = sorted(applications, key=lambda x: x["name"].lower())
            
            return {
                "applications": sorted_apps,
                "total_found": len(sorted_apps),
                "system": "Linux",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to list Linux applications: {str(e)}"}

    async def execute_system_control_with_llm(self, operation: str = None, volume: int = None, 
                                            application: str = None, llm_analysis: Dict = None) -> Dict:
        """Execute system control operations using LLM analysis"""
        
        try:
            print(f"\n🤖 **System Control Analysis**")
            print("=" * 50)
            
            if llm_analysis:
                intent = llm_analysis.get('intent', 'System operation')
                print(f"**Intent:** {intent}")
                print(f"**Operation:** {operation}")
                if volume is not None:
                    print(f"**Volume:** {volume}%")
                if application:
                    print(f"**Application:** {application}")
            
            print("=" * 50)
            
            # Execute based on operation type
            if operation == "set_volume" and volume is not None:
                print(f"\n🔊 Setting system volume to {volume}%...")
                result = await self.set_volume(volume)
                
                if result.get("success"):
                    result["message"] = f"✅ Volume successfully set to {volume}%"
                    result["llm_guided"] = True
                
                return result
            
            elif operation == "cleanup":
                print("\n🧹 Starting system cleanup...")
                
                # Clear recycle bin
                recycle_result = await self.clear_recycle_bin(confirm=True)
                
                # Clean temp files
                temp_result = await self.clean_temp_files(confirm=True)
                
                return {
                    "success": True,
                    "message": "✅ System cleanup completed successfully!",
                    "operations": {
                        "recycle_bin": recycle_result,
                        "temp_files": temp_result
                    },
                    "llm_guided": True,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif operation == "open_application" and application:
                print(f"\n🚀 Opening application: {application}...")
                result = await self.open_application(application)
                
                if result.get("success"):
                    result["message"] = f"✅ Successfully opened {application}"
                    result["llm_guided"] = True
                
                return result
            
            elif operation == "general":
                # For general operations, try to parse from the original request
                original_request = llm_analysis.get('parameters', {}).get('description', '') if llm_analysis else ''
                
                print(f"\n🔍 Analyzing general system request: {original_request}")
                
                # Try to determine what the user wants
                request_lower = original_request.lower()
                
                if any(word in request_lower for word in ['volume', 'sound']):
                    # Try to extract volume level
                    import re
                    volume_match = re.search(r'(\d+)', original_request)
                    if volume_match:
                        volume_level = int(volume_match.group(1))
                        return await self.execute_system_control_with_llm("set_volume", volume_level, None, llm_analysis)
                
                elif any(word in request_lower for word in ['clean', 'recycle', 'temp']):
                    return await self.execute_system_control_with_llm("cleanup", None, None, llm_analysis)
                
                elif any(word in request_lower for word in ['open', 'launch', 'start']):
                    # Try to extract application name
                    words = original_request.split()
                    for i, word in enumerate(words):
                        if word.lower() in ['open', 'launch', 'start'] and i + 1 < len(words):
                            app_name = words[i + 1]
                            return await self.execute_system_control_with_llm("open_application", None, app_name, llm_analysis)
                
                return {
                    "success": False,
                    "message": f"❌ Could not determine specific action for: {original_request}",
                    "suggestion": "Please be more specific about what system operation you want to perform",
                    "llm_guided": True
                }
            
            else:
                return {
                    "success": False,
                    "message": f"❌ Unknown operation: {operation}",
                    "available_operations": ["set_volume", "cleanup", "open_application"],
                    "llm_guided": True
                }
                
        except Exception as e:
            logger.error(f"Error in LLM-guided system control: {e}")
            return {
                "success": False,
                "error": f"System control failed: {str(e)}",
                "llm_guided": True
            }