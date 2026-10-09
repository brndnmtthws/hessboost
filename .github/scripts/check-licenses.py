"""Check that Python distributions carry the repository's license text.

Usage: check-licenses.py LICENSE DIST...

Apache-2.0 section 4(a) requires every copy of the work to include the
license. Each wheel and sdist must name its license files in its metadata
(``License-File``), contain each of them (a wheel under
``.dist-info/licenses/``, an sdist at its root), and include one identical
to LICENSE.
"""

import email
import sys
import tarfile
import zipfile
from pathlib import Path


def license_files(path: Path) -> dict[str, bytes | None]:
    """The license files ``path``'s metadata names, each with its contents
    (``None`` for one the distribution lacks)."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as wheel:
            names = set(wheel.namelist())
            metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
            licenses = metadata.removesuffix("METADATA") + "licenses/"
            named = email.message_from_bytes(wheel.read(metadata)).get_all("License-File") or []
            return {
                name: wheel.read(licenses + name) if licenses + name in names else None
                for name in named
            }
    with tarfile.open(path) as sdist:
        root = sdist.getnames()[0].split("/")[0]

        def read(name: str) -> bytes | None:
            try:
                member = sdist.extractfile(f"{root}/{name}")
            except KeyError:
                return None
            return None if member is None else member.read()

        info = read("PKG-INFO") or b""
        named = email.message_from_bytes(info).get_all("License-File") or []
        return {name: read(name) for name in named}


def main() -> int:
    text = Path(sys.argv[1]).read_bytes()
    failures = []
    for dist in sys.argv[2:]:
        files = license_files(Path(dist))
        if not files or None in files.values() or text not in files.values():
            found = ", ".join(f"{n}{'' if c else ' (missing)'}" for n, c in files.items())
            failures.append(f"{dist}: {found or 'no License-File'}")
    if failures:
        print("Without the repository's LICENSE:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print(f"{len(sys.argv) - 2} distributions carry the repository's LICENSE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
