#!/usr/bin/env bash
set -u
umask 022

readonly OUT="${ANTIVENOM_METADATA_DIR:-/local/antivenom/metadata}"

mkdir -p "${OUT}"

capture() {
    local name="$1"
    shift

    local output="${OUT}/${name}"
    local error="${OUT}/${name}.error"
    local temporary_error="${error}.tmp"

    if "$@" >"${output}" 2>"${error}"; then
        rm -f "${error}"
    else
        local rc=$?

        {
            echo "exit_code=${rc}"
            printf "command="
            printf "%q " "$@"
            echo
            echo
            cat "${error}" 2>/dev/null || true
        } >"${temporary_error}"

        mv "${temporary_error}" "${error}"
        return 0
    fi
}

capture_shell() {
    local name="$1"
    shift
    capture "${name}" bash -c "$*"
}

date --iso-8601=seconds >"${OUT}/collected-at.txt"
date +%s >"${OUT}/collected-at-unix.txt"
hostname >"${OUT}/hostname.txt"
capture_shell hostname-fqdn.txt 'hostname -f 2>/dev/null || hostname'
capture machine-id.txt cat /etc/machine-id

{
    echo "cluster=${ANTIVENOM_CLUSTER:-unknown}"
    echo "hardware_type=${ANTIVENOM_HARDWARE_TYPE:-unknown}"
    echo "hostname=$(hostname)"
} >"${OUT}/cloudlab-selection.txt"

capture_shell cloudlab-environment.txt '
    env | sort | grep -E "^(ANTIVENOM_|CLOUDLAB_|EMULAB_|GENI_|HOSTNAME=)" || true
'

capture os-release.txt cat /etc/os-release
capture uname.txt uname -a
capture kernel-release.txt uname -r
capture kernel-version.txt cat /proc/version
capture architecture.txt uname -m
capture kernel-cmdline.txt cat /proc/cmdline
capture uptime.txt cat /proc/uptime
capture boot-id.txt cat /proc/sys/kernel/random/boot_id
capture_shell boot-time.txt 'uptime -s 2>/dev/null || who -b 2>/dev/null || true'

capture lscpu.txt lscpu
capture lscpu.json lscpu -J
capture cpuinfo.txt cat /proc/cpuinfo
capture_shell cpu-topology.txt '
    lscpu -e=CPU,NODE,SOCKET,CORE,CACHE,ONLINE,MAXMHZ,MINMHZ 2>/dev/null ||
        lscpu -e=CPU,NODE,SOCKET,CORE,CACHE
'
capture_shell cpu-summary.txt '
    echo "architecture=$(uname -m)"
    awk -F: "
        /^vendor_id/ && !vendor { gsub(/^[ \t]+/, \"\", \$2); vendor=\$2 }
        /^model name/ && !model_name { gsub(/^[ \t]+/, \"\", \$2); model_name=\$2 }
        /^cpu family/ && !family { gsub(/^[ \t]+/, \"\", \$2); family=\$2 }
        /^model[ \t]*:/ && !model_number { gsub(/^[ \t]+/, \"\", \$2); model_number=\$2 }
        /^stepping/ && !stepping { gsub(/^[ \t]+/, \"\", \$2); stepping=\$2 }
        /^microcode/ && !microcode { gsub(/^[ \t]+/, \"\", \$2); microcode=\$2 }
        END {
            print \"vendor_id=\" vendor
            print \"model_name=\" model_name
            print \"cpu_family=\" family
            print \"model=\" model_number
            print \"stepping=\" stepping
            print \"microcode=\" microcode
        }
    " /proc/cpuinfo
    echo "logical_cpus=$(nproc --all)"
    echo "available_cpus=$(nproc)"
'
capture_shell microcode.txt '
    grep -m1 "^microcode" /proc/cpuinfo || true
    if command -v dmesg >/dev/null 2>&1; then
        dmesg 2>/dev/null | grep -i microcode | head -n 50 || true
    fi
'

capture_shell cpu-state.txt '
    for file in \
        /sys/devices/system/cpu/possible \
        /sys/devices/system/cpu/present \
        /sys/devices/system/cpu/online \
        /sys/devices/system/cpu/offline \
        /sys/devices/system/cpu/isolated \
        /sys/devices/system/cpu/nohz_full; do
        if [[ -r "$file" ]]; then
            printf "%s=" "$file"
            cat "$file"
        fi
    done
    if [[ -r /sys/devices/system/cpu/smt/active ]]; then
        echo "smt_active=$(cat /sys/devices/system/cpu/smt/active)"
    fi
    if [[ -r /sys/devices/system/cpu/smt/control ]]; then
        echo "smt_control=$(cat /sys/devices/system/cpu/smt/control)"
    fi
'
capture process-affinity.txt taskset -pc "$$"

capture_shell cpufreq-policy.txt '
    found=0
    for policy in /sys/devices/system/cpu/cpufreq/policy*; do
        [[ -d "$policy" ]] || continue
        found=1
        echo "[$policy]"
        for field in \
            affected_cpus related_cpus scaling_driver scaling_governor \
            scaling_available_governors scaling_available_frequencies \
            scaling_cur_freq scaling_min_freq scaling_max_freq \
            cpuinfo_cur_freq cpuinfo_min_freq cpuinfo_max_freq \
            energy_performance_available_preferences energy_performance_preference; do
            if [[ -r "$policy/$field" ]]; then
                printf "%s=" "$field"
                cat "$policy/$field"
            fi
        done
        echo
    done
    if [[ "$found" -eq 0 ]]; then
        echo "cpufreq policy information unavailable"
    fi
'
capture_shell cpu-turbo-boost.txt '
    files=(
        /sys/devices/system/cpu/intel_pstate/no_turbo
        /sys/devices/system/cpu/intel_pstate/status
        /sys/devices/system/cpu/intel_pstate/max_perf_pct
        /sys/devices/system/cpu/intel_pstate/min_perf_pct
        /sys/devices/system/cpu/cpufreq/boost
        /sys/devices/system/cpu/amd_pstate/status
    )
    found=0
    for file in "${files[@]}"; do
        if [[ -r "$file" ]]; then
            found=1
            printf "%s=" "$file"
            cat "$file"
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "turbo/boost information unavailable"
    fi
'
capture_shell current-frequencies.txt '
    found=0
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        [[ -d "$cpu" ]] || continue
        echo "[$(basename "$cpu")]"
        for file in "$cpu/cpufreq/scaling_cur_freq" "$cpu/cpufreq/cpuinfo_cur_freq"; do
            if [[ -r "$file" ]]; then
                found=1
                printf "%s=" "$(basename "$file")"
                cat "$file"
            fi
        done
    done
    if [[ "$found" -eq 0 ]]; then
        echo "current CPU frequency information unavailable"
    fi
'
if command -v cpupower >/dev/null 2>&1; then
    capture cpupower-frequency-info.txt cpupower frequency-info
fi

capture_shell cache-sysfs.txt '
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        [[ -d "$cpu/cache" ]] || continue
        echo "=== $(basename "$cpu") ==="
        for index in "$cpu"/cache/index*; do
            [[ -d "$index" ]] || continue
            echo "[$index]"
            for field in level type size coherency_line_size number_of_sets \
                ways_of_associativity physical_line_partition shared_cpu_list shared_cpu_map; do
                if [[ -r "$index/$field" ]]; then
                    printf "%s=" "$field"
                    cat "$index/$field"
                fi
            done
        done
        echo
    done
'

capture memory-proc.txt cat /proc/meminfo
capture free-bytes.txt free -b
if command -v numactl >/dev/null 2>&1; then
    capture numa.txt numactl --hardware
fi
capture_shell numa-sysfs.txt '
    found=0
    for node in /sys/devices/system/node/node*; do
        [[ -d "$node" ]] || continue
        found=1
        echo "=== $(basename "$node") ==="
        [[ -r "$node/cpulist" ]] && echo "cpulist=$(cat "$node/cpulist")"
        [[ -r "$node/distance" ]] && echo "distance=$(cat "$node/distance")"
        [[ -r "$node/meminfo" ]] && cat "$node/meminfo"
        echo
    done
    if [[ "$found" -eq 0 ]]; then
        echo "NUMA sysfs information unavailable"
    fi
'
if command -v dmidecode >/dev/null 2>&1; then
    capture dmi-memory.txt dmidecode -t memory
    capture dmi-system.txt dmidecode -t system
    capture dmi-bios.txt dmidecode -t bios
    capture dmi-baseboard.txt dmidecode -t baseboard
    capture dmi-processor.txt dmidecode -t processor
    capture dmi-chassis.txt dmidecode -t chassis
    capture_shell system-summary.txt '
        fields=(
            system-manufacturer system-product-name system-version system-serial-number
            baseboard-manufacturer baseboard-product-name baseboard-version bios-vendor
            bios-version bios-release-date chassis-manufacturer chassis-type
        )
        for field in "${fields[@]}"; do
            value=$(dmidecode -s "$field" 2>/dev/null || true)
            printf "%s=%s\n" "$field" "$value"
        done
    '
fi

capture_shell virtualization.txt '
    if command -v systemd-detect-virt >/dev/null 2>&1; then
        value=$(systemd-detect-virt 2>/dev/null)
        rc=$?
        if [[ "$rc" -eq 0 ]]; then
            echo "$value"
        elif [[ "$rc" -eq 1 ]]; then
            echo "none"
        else
            echo "unknown"
        fi
    else
        echo "systemd-detect-virt unavailable"
    fi
    echo
    lscpu | grep -E "Hypervisor vendor|Virtualization type|Virtualization:" || true
'

capture cgroup-self.txt cat /proc/self/cgroup
capture proc-limits.txt cat /proc/self/limits
capture limits.txt bash -c 'ulimit -a'
if command -v findmnt >/dev/null 2>&1; then
    capture mounts.txt findmnt
fi
capture mount-table.txt cat /proc/mounts
capture_shell cgroup-details.txt '
    echo "=== cgroup mounts ==="
    mount | grep cgroup || true
    echo
    echo "=== current cgroup ==="
    cgroup_path=$(awk -F: '\''$1 == "0" {print $3}'\'' /proc/self/cgroup)
    if [[ -n "$cgroup_path" && -d "/sys/fs/cgroup${cgroup_path}" ]]; then
        directory="/sys/fs/cgroup${cgroup_path}"
        for file in cpu.max cpu.weight cpuset.cpus cpuset.cpus.effective \
            cpuset.mems cpuset.mems.effective memory.max memory.high \
            memory.current memory.swap.max pids.max pids.current; do
            if [[ -r "$directory/$file" ]]; then
                printf "%s=" "$file"
                cat "$directory/$file"
            fi
        done
    else
        echo "cgroup v2 directory unavailable"
    fi
'
capture namespaces.txt ls -l /proc/self/ns

if command -v lspci >/dev/null 2>&1; then
    capture pci.txt lspci -nn
    capture pci-kernel-drivers.txt lspci -nnk
    capture pci-tree.txt lspci -tv
fi
capture_shell pci-numa.txt '
    for device_path in /sys/bus/pci/devices/*; do
        [[ -d "$device_path" ]] || continue
        device_name=$(basename "$device_path")
        vendor=$(cat "$device_path/vendor" 2>/dev/null || true)
        device=$(cat "$device_path/device" 2>/dev/null || true)
        class=$(cat "$device_path/class" 2>/dev/null || true)
        numa_node=$(cat "$device_path/numa_node" 2>/dev/null || true)
        local_cpulist=$(cat "$device_path/local_cpulist" 2>/dev/null || true)
        printf "%s vendor=%s device=%s class=%s numa_node=%s local_cpulist=%s\n" \
            "$device_name" "$vendor" "$device" "$class" "$numa_node" "$local_cpulist"
    done
'

if command -v ip >/dev/null 2>&1; then
    capture network-links.txt ip -details link show
    capture network-addresses.txt ip address show
    capture network-routes.txt ip route show table all
    capture network-rules.txt ip rule show
fi
if command -v ss >/dev/null 2>&1; then
    capture socket-summary.txt ss -s
fi
capture_shell network-drivers.txt '
    for interface_path in /sys/class/net/*; do
        [[ -e "$interface_path" ]] || continue
        interface=$(basename "$interface_path")
        [[ "$interface" == "lo" ]] && continue
        echo "=== $interface ==="
        if command -v ethtool >/dev/null 2>&1; then
            echo "--- driver ---"; ethtool -i "$interface" 2>&1 || true
            echo "--- link ---"; ethtool "$interface" 2>&1 || true
            echo "--- offloads ---"; ethtool -k "$interface" 2>&1 || true
            echo "--- channels ---"; ethtool -l "$interface" 2>&1 || true
            echo "--- rings ---"; ethtool -g "$interface" 2>&1 || true
        else
            echo "ethtool unavailable"
        fi
        echo
    done
'
capture_shell network-numa.txt '
    for interface_path in /sys/class/net/*; do
        [[ -e "$interface_path" ]] || continue
        interface=$(basename "$interface_path")
        [[ "$interface" == "lo" ]] && continue
        echo "=== $interface ==="
        device_path=$(readlink -f "$interface_path/device" 2>/dev/null || true)
        echo "device=$device_path"
        if [[ -n "$device_path" ]]; then
            [[ -r "$device_path/numa_node" ]] && echo "numa_node=$(cat "$device_path/numa_node")"
            [[ -r "$device_path/local_cpulist" ]] && echo "local_cpulist=$(cat "$device_path/local_cpulist")"
        fi
        echo
    done
'

if command -v lsblk >/dev/null 2>&1; then
    capture block-devices.txt lsblk -O
    capture_shell block-topology.txt '
        lsblk -o NAME,KNAME,TYPE,SIZE,ROTA,RO,MOUNTPOINT,FSTYPE,MODEL,VENDOR,SERIAL,TRAN,PHY-SEC,LOG-SEC,SCHED,HCTL 2>/dev/null ||
            lsblk -o NAME,KNAME,TYPE,SIZE,ROTA,RO,MOUNTPOINT,FSTYPE,MODEL,VENDOR,SERIAL
    '
fi
capture filesystems.txt df -T
capture disk-usage.txt df -B1
capture_shell storage-details.txt '
    for device_path in /sys/block/*; do
        [[ -d "$device_path" ]] || continue
        device=$(basename "$device_path")
        echo "=== $device ==="
        for file in \
            "$device_path/device/vendor" "$device_path/device/model" \
            "$device_path/device/rev" "$device_path/queue/scheduler" \
            "$device_path/queue/rotational" "$device_path/queue/logical_block_size" \
            "$device_path/queue/physical_block_size" "$device_path/queue/nr_requests" \
            "$device_path/queue/read_ahead_kb"; do
            if [[ -r "$file" ]]; then
                printf "%s=" "${file#"${device_path}/"}"
                cat "$file"
            fi
        done
        echo
    done
'

capture interrupts.txt cat /proc/interrupts
capture softirqs.txt cat /proc/softirqs
capture_shell irq-affinity.txt '
    for irq_path in /proc/irq/[0-9]*; do
        [[ -d "$irq_path" ]] || continue
        irq=$(basename "$irq_path")
        affinity=$(cat "$irq_path/smp_affinity_list" 2>/dev/null || true)
        printf "%s %s\n" "$irq" "$affinity"
    done
'
capture_shell irqbalance.txt '
    if command -v systemctl >/dev/null 2>&1 &&
       systemctl list-unit-files irqbalance.service >/dev/null 2>&1; then
        systemctl status irqbalance --no-pager 2>&1 || true
    else
        echo "irqbalance service unavailable"
    fi
'

if command -v sensors >/dev/null 2>&1; then
    capture sensors.txt sensors
fi
capture_shell thermal-zones.txt '
    found=0
    for zone in /sys/class/thermal/thermal_zone*; do
        [[ -d "$zone" ]] || continue
        found=1
        echo "[$zone]"
        [[ -r "$zone/type" ]] && echo "type=$(cat "$zone/type")"
        [[ -r "$zone/temp" ]] && echo "temp=$(cat "$zone/temp")"
        echo
    done
    if [[ "$found" -eq 0 ]]; then
        echo "thermal zones unavailable"
    fi
'
capture_shell hwmon.txt '
    found=0
    for hwmon in /sys/class/hwmon/hwmon*; do
        [[ -d "$hwmon" ]] || continue
        found=1
        echo "=== $hwmon ==="
        [[ -r "$hwmon/name" ]] && echo "name=$(cat "$hwmon/name")"
        for file in "$hwmon"/*_input "$hwmon"/*_label; do
            [[ -r "$file" ]] || continue
            printf "%s=" "$(basename "$file")"
            cat "$file"
        done
        echo
    done
    if [[ "$found" -eq 0 ]]; then
        echo "hwmon information unavailable"
    fi
'
capture_shell powercap.txt '
    found=0
    for zone in /sys/class/powercap/*; do
        [[ -d "$zone" ]] || continue
        found=1
        echo "=== $zone ==="
        for file in name enabled energy_uj max_energy_range_uj constraint_0_name \
            constraint_0_power_limit_uw constraint_0_time_window_us; do
            if [[ -r "$zone/$file" ]]; then
                printf "%s=" "$file"
                cat "$zone/$file"
            fi
        done
        echo
    done
    if [[ "$found" -eq 0 ]]; then
        echo "powercap information unavailable"
    fi
'

capture loaded-modules.txt lsmod
capture sysctl.txt sysctl -a
capture_shell performance-sysctl.txt '
    keys=(
        kernel.perf_event_paranoid kernel.kptr_restrict kernel.nmi_watchdog
        kernel.sched_autogroup_enabled kernel.sched_migration_cost_ns
        kernel.sched_min_granularity_ns kernel.sched_wakeup_granularity_ns
        vm.swappiness vm.overcommit_memory vm.dirty_ratio vm.dirty_background_ratio
        vm.zone_reclaim_mode net.core.rmem_max net.core.wmem_max net.core.netdev_max_backlog
    )
    for key in "${keys[@]}"; do
        value=$(sysctl -n "$key" 2>/dev/null || true)
        printf "%s=%s\n" "$key" "$value"
    done
'

capture dmesg.txt dmesg --ctime
capture_shell dmesg-warnings.txt '
    dmesg --ctime 2>/dev/null |
        grep -iE "error|fail|fault|mce|edac|thermal|throttl|timeout|reset|watchdog|corrected|uncorrected" || true
'
capture_shell journal-kernel.txt '
    if command -v journalctl >/dev/null 2>&1; then
        journalctl -k --no-pager 2>/dev/null || true
    else
        echo "journalctl unavailable"
    fi
'

if command -v perf >/dev/null 2>&1; then
    capture perf-version.txt perf --version
    capture perf-list.txt perf list
    capture_shell perf-permissions.txt '
        echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || true)"
        echo "kptr_restrict=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null || true)"
        if [[ -r /proc/sys/kernel/nmi_watchdog ]]; then
            echo "nmi_watchdog=$(cat /proc/sys/kernel/nmi_watchdog)"
        fi
    '
else
    echo "perf unavailable" >"${OUT}/perf-unavailable.txt"
fi

capture process-list.txt ps auxww
capture process-tree.txt ps -eF --forest
capture loadavg.txt cat /proc/loadavg
capture vmstat-baseline.txt vmstat -s
capture_shell systemd-failed.txt '
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --failed --no-pager 2>/dev/null || true
    else
        echo "systemctl unavailable"
    fi
'

if command -v timedatectl >/dev/null 2>&1; then
    capture timedatectl.txt timedatectl
fi
capture_shell clocksource.txt '
    for file in \
        /sys/devices/system/clocksource/clocksource0/current_clocksource \
        /sys/devices/system/clocksource/clocksource0/available_clocksource; do
        if [[ -r "$file" ]]; then
            printf "%s=" "$file"
            cat "$file"
        fi
    done
'
capture_shell ntp-status.txt '
    if command -v timedatectl >/dev/null 2>&1; then
        timedatectl show -p NTPSynchronized -p NTP -p TimeUSec -p Timezone 2>/dev/null || true
    else
        echo "timedatectl unavailable"
    fi
'

capture_shell metadata-checksums.txt "
    find '${OUT}' \\
        -maxdepth 1 \\
        -type f \\
        ! -name metadata-checksums.txt \\
        ! -name COLLECTION_COMPLETE \\
        -print0 |
        sort -z |
        xargs -0 -r sha256sum
"

{
    echo "status=complete"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "output_directory=${OUT}"
} >"${OUT}/COLLECTION_COMPLETE"

echo "Metadata collection complete: ${OUT}"
