#!/usr/bin/env python3
"""Install and manage the K7C AP + ball tracker systemd service."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


SERVICE_NAME = "k7c-ball.service"
SERVICE_SOURCE = Path(__file__).with_name(SERVICE_NAME)
SERVICE_TARGET = Path("/etc/systemd/system") / SERVICE_NAME
TRACKER = Path("/userdata/project/grayscale_ball_tracker_rk3576.py")
VENV_PYTHON = Path("/venv/bin/python3")
RECORD_DIRECTORY = Path("/userdata/recordings")

WIFI_INTERFACE = "wlan0"
AP_CONNECTION = "K7C-AP"
AP_SSID = "K7C-AP"
AP_PASSWORD = "YOUR_WIFI_PASSWORD"
AP_ADDRESS = "10.42.0.1/24"


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    display_command = list(command)
    try:
        psk_index = display_command.index("802-11-wireless-security.psk")
        display_command[psk_index + 1] = "********"
    except (ValueError, IndexError):
        pass
    print("+", " ".join(display_command))
    return subprocess.run(command, check=check, text=True)


def require_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit("Run this command as root, for example: sudo python3 ...")


def check_required_files() -> None:
    missing = [
        str(path)
        for path in (SERVICE_SOURCE, TRACKER, VENV_PYTHON)
        if not path.exists()
    ]
    if missing:
        raise SystemExit("Missing required file(s):\n  " + "\n  ".join(missing))
    if shutil.which("nmcli") is None:
        raise SystemExit("nmcli is missing; install/enable NetworkManager first")


def connection_exists(name: str) -> bool:
    result = subprocess.run(
        ["nmcli", "-t", "-f", "NAME", "connection", "show"],
        check=True,
        capture_output=True,
        text=True,
    )
    return name in result.stdout.splitlines()


def configure_hotspot() -> None:
    run(["nmcli", "radio", "wifi", "on"])
    if not connection_exists(AP_CONNECTION):
        run(
            [
                "nmcli",
                "connection",
                "add",
                "type",
                "wifi",
                "ifname",
                WIFI_INTERFACE,
                "con-name",
                AP_CONNECTION,
                "ssid",
                AP_SSID,
            ]
        )

    run(
        [
            "nmcli",
            "connection",
            "modify",
            AP_CONNECTION,
            "connection.interface-name",
            WIFI_INTERFACE,
            "connection.autoconnect",
            "no",
            "802-11-wireless.mode",
            "ap",
            "802-11-wireless.band",
            "bg",
            "802-11-wireless.channel",
            "6",
            "802-11-wireless.powersave",
            "2",
            "ipv4.method",
            "shared",
            "ipv4.addresses",
            AP_ADDRESS,
            "ipv6.method",
            "disabled",
            "802-11-wireless-security.key-mgmt",
            "wpa-psk",
            "802-11-wireless-security.proto",
            "rsn",
            "802-11-wireless-security.psk",
            AP_PASSWORD,
        ]
    )


def active_connection_exists(name: str) -> bool:
    result = subprocess.run(
        ["nmcli", "-t", "-f", "NAME", "connection", "show", "--active"],
        check=True,
        capture_output=True,
        text=True,
    )
    return name in result.stdout.splitlines()


def ensure_ap() -> None:
    require_root()
    if not connection_exists(AP_CONNECTION):
        configure_hotspot()
    else:
        run(["nmcli", "radio", "wifi", "on"])
    if not active_connection_exists(AP_CONNECTION):
        run(
            [
                "nmcli",
                "connection",
                "up",
                AP_CONNECTION,
                "ifname",
                WIFI_INTERFACE,
            ]
        )


def install() -> None:
    require_root()
    check_required_files()
    configure_hotspot()
    RECORD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SERVICE_SOURCE, SERVICE_TARGET)
    os.chmod(SERVICE_TARGET, 0o644)
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "enable", SERVICE_NAME])
    print()
    print("Installed and enabled. The current Wi-Fi connection was left untouched.")
    print("Reboot to start the AP and tracker: sudo reboot")
    print("To switch immediately (the current SSH session will disconnect):")
    print("  sudo python3 %s start" % Path(__file__).resolve())


def start() -> None:
    require_root()
    print("WARNING: wlan0 will switch to AP mode and may disconnect this terminal.")
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", SERVICE_NAME])


def stop() -> None:
    require_root()
    run(["systemctl", "stop", SERVICE_NAME], check=False)
    run(["nmcli", "connection", "down", AP_CONNECTION], check=False)


def status() -> None:
    run(["systemctl", "status", SERVICE_NAME, "--no-pager", "-l"], check=False)
    run(["nmcli", "device", "status"], check=False)
    run(["ip", "-4", "address", "show", WIFI_INTERFACE], check=False)


def logs() -> None:
    run(["journalctl", "-u", SERVICE_NAME, "-n", "100", "--no-pager"], check=False)


def uninstall() -> None:
    require_root()
    run(["systemctl", "disable", "--now", SERVICE_NAME], check=False)
    if SERVICE_TARGET.exists():
        SERVICE_TARGET.unlink()
    run(["systemctl", "daemon-reload"])
    print("Service removed. The NetworkManager profile K7C-AP was preserved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "install",
            "ensure-ap",
            "start",
            "stop",
            "status",
            "logs",
            "uninstall",
        ),
    )
    return parser.parse_args()


def main() -> None:
    actions = {
        "install": install,
        "ensure-ap": ensure_ap,
        "start": start,
        "stop": stop,
        "status": status,
        "logs": logs,
        "uninstall": uninstall,
    }
    actions[parse_args().action]()


if __name__ == "__main__":
    main()
