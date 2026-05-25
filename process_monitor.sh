#!/bin/bash

# 进程监控脚本
# 功能：监控指定PID列表的进程总内存占比，超过阈值则kill所有进程
# 监控间隔：1秒检查一次
# 日志输出：每300秒输出一次占比信息
# 用法：./monitor_processes.sh

# ==================== 配置参数 ====================
# PID数组（请修改为您需要监控的进程ID）
PIDS=(1739645 1739646)

# 内存阈值（百分比）
THRESHOLD=20

# 监控间隔（秒）- 实际检查频率
MONITOR_INTERVAL=1

# 日志输出间隔（秒）- 每300秒输出一次日志
LOG_INTERVAL=300

# 日志文件（可选，留空则不记录日志）
LOG_FILE="./process_monitor.log"

# ==================== 函数定义 ====================

# 获取系统总内存（KB）
get_total_mem() {
    # 方法1：从/proc/meminfo获取
    if [[ -f /proc/meminfo ]]; then
        grep MemTotal /proc/meminfo | awk '{print $2}'
    # 方法2：使用free命令（备用）
    else
        free -k | awk '/^Mem:/ {print $2}'
    fi
}

# 获取单个进程的内存使用（KB）
# 参数：pid
get_process_mem() {
    local pid=$1
    local mem=0
    
    # 检查进程是否存在
    if [[ -d /proc/$pid ]]; then
        # 从/proc/$pid/statm获取：第1项是总大小，第2项是RSS（单位：页）
        # 通常页大小为4KB，使用statm第2项（RSS）
        if [[ -f /proc/$pid/statm ]]; then
            mem=$(awk '{print $2}' /proc/$pid/statm 2>/dev/null)
            # 转换为KB（通常页大小为4KB，根据系统调整）
            mem=$((mem * 4))
        fi
        
        # 如果statm失败，尝试从smaps累加（更准确但较慢）
        if [[ -z "$mem" || "$mem" -eq 0 ]]; then
            mem=$(awk '/^Pss:/ {sum+=$2} END {print sum}' /proc/$pid/smaps 2>/dev/null)
        fi
    fi
    
    echo "${mem:-0}"
}

# 计算所有指定进程的总内存占比
# 参数：pid数组
calculate_total_mem_percentage() {
    local total_mem_kb=$(get_total_mem)
    local total_process_mem_kb=0
    local alive_pids=()
    
    # 遍历所有PID
    for pid in "${PIDS[@]}"; do
        # 检查进程是否存在
        if kill -0 "$pid" 2>/dev/null; then
            local mem=$(get_process_mem "$pid")
            total_process_mem_kb=$((total_process_mem_kb + mem))
            alive_pids+=("$pid")
        fi
    done
    
    # 更新全局PIDS数组为实际存活的进程
    PIDS=("${alive_pids[@]}")
    
    # 计算百分比
    if [[ $total_mem_kb -gt 0 ]] && [[ ${#PIDS[@]} -gt 0 ]]; then
        # 使用bc进行浮点计算（如果没有bc，使用整数计算）
        if command -v bc &> /dev/null; then
            percentage=$(echo "scale=2; $total_process_mem_kb * 100 / $total_mem_kb" | bc)
        else
            percentage=$((total_process_mem_kb * 100 / total_mem_kb))
        fi
        echo "$percentage"
    else
        echo "0"
    fi
}

# 终止所有监控的进程
kill_all_processes() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] ========== 开始终止所有监控进程 =========="
    [[ -n "$LOG_FILE" ]] && echo "[$timestamp] ========== 开始终止所有监控进程 ==========" >> "$LOG_FILE"
    
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[$timestamp] 终止进程 PID: $pid"
            [[ -n "$LOG_FILE" ]] && echo "[$timestamp] 终止进程 PID: $pid" >> "$LOG_FILE"
            
            # 先尝试优雅终止（SIGTERM）
            kill -15 "$pid" 2>/dev/null
            
            # 等待2秒，如果还在则强制终止（SIGKILL）
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                echo "[$timestamp] 强制终止进程 PID: $pid"
                [[ -n "$LOG_FILE" ]] && echo "[$timestamp] 强制终止进程 PID: $pid" >> "$LOG_FILE"
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done
    
    echo "[$timestamp] ========== 所有进程已终止 =========="
    [[ -n "$LOG_FILE" ]] && echo "[$timestamp] ========== 所有进程已终止 ==========" >> "$LOG_FILE"
}

# 打印使用帮助
print_usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -p, --pids PID1,PID2,...  指定要监控的进程ID（逗号分隔）"
    echo "  -t, --threshold PERCENT    设置内存阈值（百分比，默认: 20）"
    echo "  -m, --monitor-interval SEC  设置监控间隔秒数（默认: 1秒）"
    echo "  -l, --log-interval SEC     设置日志输出间隔秒数（默认: 300秒）"
    echo "  -L, --log-file FILE        指定日志文件路径"
    echo "  -d, --daemon               以后台守护进程模式运行"
    echo "  -h, --help                 显示此帮助信息"
    echo ""
    echo "说明:"
    echo "  - 监控间隔默认为1秒，实时检查内存使用情况"
    echo "  - 日志输出间隔默认300秒，减少日志输出频率"
    echo ""
    echo "示例:"
    echo "  $0 -p 1234,5678,9012 -t 25 -m 1 -l 300"
    echo "  $0 --pids 1234,5678 --threshold 30 --log-interval 600"
    echo "  $0 -p 1234,5678 -t 20 -L /var/log/monitor.log -d"
}

# ==================== 主程序 ====================

# 解析命令行参数
DAEMON_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pids)
            IFS=',' read -ra PIDS <<< "$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -m|--monitor-interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        -l|--log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        -L|--log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -d|--daemon)
            DAEMON_MODE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            print_usage
            exit 1
            ;;
    esac
done

# 验证参数
if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "错误: 未指定要监控的进程ID"
    print_usage
    exit 1
fi

if ! [[ "$THRESHOLD" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "错误: 阈值必须是数字"
    exit 1
fi

if ! [[ "$MONITOR_INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "错误: 监控间隔必须是整数"
    exit 1
fi

if ! [[ "$LOG_INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "错误: 日志间隔必须是整数"
    exit 1
fi

# 检查bc命令（可选）
if ! command -v bc &> /dev/null; then
    echo "警告: bc命令未安装，将使用整数计算（精度可能较低）"
fi

# 检查是否有权限访问/proc
if [[ ! -d /proc ]]; then
    echo "错误: 无法访问/proc目录，请确保脚本在Linux系统上运行"
    exit 1
fi

# 以后台守护进程模式运行
if [[ "$DAEMON_MODE" == true ]]; then
    echo "启动守护进程模式，PID: $$"
    echo "监控进程: ${PIDS[*]}"
    echo "内存阈值: ${THRESHOLD}%"
    echo "监控间隔: ${MONITOR_INTERVAL}秒（实时检查）"
    echo "日志间隔: ${LOG_INTERVAL}秒（每${LOG_INTERVAL}秒输出一次）"
    echo "日志文件: ${LOG_FILE:-未启用}"
    
    # 脱离终端并后台运行
    setsid "$0" --pids "$(IFS=,; echo "${PIDS[*]}")" \
                 --threshold "$THRESHOLD" \
                 --monitor-interval "$MONITOR_INTERVAL" \
                 --log-interval "$LOG_INTERVAL" \
                 ${LOG_FILE:+--log-file "$LOG_FILE"} \
                 > /dev/null 2>&1 < /dev/null &
    exit 0
fi

# 主循环变量
log_counter=0
check_count=0
start_time=$(date +%s)

echo "=========================================="
echo "进程监控脚本已启动"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "监控进程: ${PIDS[*]}"
echo "内存阈值: ${THRESHOLD}%"
echo "监控间隔: ${MONITOR_INTERVAL}秒（每${MONITOR_INTERVAL}秒检查一次）"
echo "日志间隔: ${LOG_INTERVAL}秒（每${LOG_INTERVAL}秒输出一次状态）"
echo "日志文件: ${LOG_FILE:-未启用}"
echo "=========================================="

# 写入启动日志
if [[ -n "$LOG_FILE" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 监控脚本启动 ==========" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控进程列表: ${PIDS[*]}" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 内存阈值: ${THRESHOLD}%" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控间隔: ${MONITOR_INTERVAL}秒" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 日志间隔: ${LOG_INTERVAL}秒" >> "$LOG_FILE"
fi

while true; do
    # 检查是否还有存活的进程
    alive_count=0
    alive_pids_temp=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((alive_count++))
            alive_pids_temp+=("$pid")
        fi
    done
    
    # 更新PID列表（移除已退出的进程）
    if [[ ${#alive_pids_temp[@]} -ne ${#PIDS[@]} ]]; then
        PIDS=("${alive_pids_temp[@]}")
        if [[ -n "$LOG_FILE" ]] && [[ ${#PIDS[@]} -lt ${#alive_pids_temp[@]} ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检测到进程退出，剩余存活进程: ${#PIDS[@]}" >> "$LOG_FILE"
        fi
    fi
    
    # 如果所有进程都已退出，终止监控
    if [[ $alive_count -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有监控进程已退出，脚本终止"
        [[ -n "$LOG_FILE" ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有监控进程已退出，脚本终止" >> "$LOG_FILE"
        break
    fi
    
    # 计算总内存占比
    percentage=$(calculate_total_mem_percentage)
    
    # 每LOG_INTERVAL秒输出一次日志
    if (( log_counter >= LOG_INTERVAL )); then
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] 检查次数: $check_count, 存活进程数: ${#PIDS[@]}, 总内存占比: ${percentage}%"
        
        if [[ -n "$LOG_FILE" ]]; then
            echo "[$timestamp] 检查次数: $check_count, 存活进程数: ${#PIDS[@]}, 总内存占比: ${percentage}%" >> "$LOG_FILE"
        fi
        
        # 重置计数器
        log_counter=0
    fi
    
    # 实时检查是否超过阈值（每次检查都判断，但只在超过时输出）
    if (( $(echo "$percentage > $THRESHOLD" | bc -l 2>/dev/null || echo "$percentage > $THRESHOLD" | awk '{print ($1 > $2)}') )); then
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] 警告: 总内存占比 ${percentage}% 超过阈值 ${THRESHOLD}%，准备终止所有进程"
        echo "[$timestamp] 检查次数: $check_count, 存活进程: ${#PIDS[@]}, 内存占比: ${percentage}%"
        
        if [[ -n "$LOG_FILE" ]]; then
            echo "[$timestamp] 警告: 总内存占比 ${percentage}% 超过阈值 ${THRESHOLD}%，准备终止所有进程" >> "$LOG_FILE"
            echo "[$timestamp] 检查次数: $check_count, 存活进程: ${#PIDS[@]}, 内存占比: ${percentage}%" >> "$LOG_FILE"
        fi
        
        kill_all_processes
        break
    fi
    
    # 更新计数器和检查次数
    log_counter=$((log_counter + MONITOR_INTERVAL))
    check_count=$((check_count + 1))
    
    # 等待下一次检查（1秒间隔）
    sleep "$MONITOR_INTERVAL"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 脚本执行完毕"
[[ -n "$LOG_FILE" ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] 脚本执行完毕" >> "$LOG_FILE"