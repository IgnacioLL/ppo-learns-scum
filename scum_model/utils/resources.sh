#!/bin/bash

# Output file
log_file="system_stats_$(date +%Y%m%d_%H%M%S).log"

# Function to get GPU stats if nvidia-smi is available
get_gpu_stats() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    else
        echo "No NVIDIA GPU found"
    fi
}

# Header for the log file
echo "=== System Performance Statistics ==="
echo "Started at: $(date)" 
echo "==================================" 

echo -e "\n=== $(date) ===" 

# CPU usage and load average
echo "CPU Usage:" 
mpstat 1 1 | grep -A 5 "%usr" 
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')" 

# Memory usage
echo -e "\nMemory Usage:" 
free -m | grep -v + 

# Disk usage
echo -e "\nDisk Usage:" 
df -h | grep -v tmp 

# IO stats
echo -e "\nIO Statistics:" 
iostat -x 1 1 | grep -A 2 "avg-cpu" 

# Network stats
echo -e "\nNetwork Statistics:" 
netstat -i | head -n2 

# GPU stats if available
echo -e "\nGPU Statistics:" 
nvidia-smi 