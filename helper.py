import platform


def getOS() -> str:
    """
    Get the os type in lower case.
    :return: str, os type, value in [windows, linux, macos, freebsd, unknown].
    """
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Linux":
        return "linux"
    elif system == "Darwin":
        return "macos"
    elif system == "FreeBSD":
        return "freebsd"
    else:
        return "unknown"


def getArch() -> str:
    """
    Get the machine architecture.
    :return: str, value in [x86_64, x86, aarch64, arm32, riscv64, s390x, ppc64le, mips64, unknown].
    """
    arch = platform.machine()
    if arch == "x86_64" or arch == "AMD64" or arch == "amd64":
        return "x86_64"
    elif arch == "i386" or arch == "i686" or arch == "x86":
        return "x86"
    elif arch == "aarch64" or arch == "arm64":
        return "aarch64"
    elif arch.find("arm") != -1:
        return "arm32"
    elif arch == "riscv64":
        return "riscv64"
    elif arch == "s390x":
        return "s390x"
    elif arch == "ppc64le":
        return "ppc64le"
    elif arch == "mips64":
        return "mips64"
    else:
        return "unknown"

