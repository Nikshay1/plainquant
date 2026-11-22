# PyInstaller recipe for the desktop app. Build from the repo root:
#   pyinstaller packaging/plainquant.spec
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas = collect_data_files("plainquant")
binaries = []
hiddenimports = collect_submodules("plainquant") + collect_submodules("uvicorn")
for pkg in ("curl_cffi", "webview"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

linux = sys.platform.startswith("linux")
excludes = ["tkinter", "matplotlib", "IPython", "pytest", "PIL"]
if not linux:
    excludes += ["PyQt6", "PyQt5", "PySide6", "qtpy"]

a = Analysis(
    ["launcher.py"],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=excludes,
)
if linux:
    # Qt brings ~80 libraries and every QML module; the web view needs about 20 of them.
    import re

    KEEP_QT = {
        "Core", "DBus", "EglFSDeviceIntegration", "Gui", "Network", "OpenGL", "OpenGLWidgets", "Positioning",
        "PrintSupport", "Qml", "QmlMeta", "QmlModels", "QmlWorkerScript", "Quick", "QuickWidgets", "Svg",
        "WaylandClient", "WlShellIntegration", "XcbQpa", "WebChannel", "WebEngineCore", "WebEngineWidgets", "Widgets",
    }

    def wanted(dest):
        d = dest.replace("\\", "/")
        if "/Qt6/qml/" in d:
            return False
        if "/Qt6/translations/" in d:
            return d.endswith("qtwebengine_locales/en-US.pak")
        m = re.search(r"libQt6(\w+?)\.so", d)
        return not m or m.group(1) in KEEP_QT

    a.binaries = [x for x in a.binaries if wanted(x[0])]
    a.datas = [x for x in a.datas if wanted(x[0])]

pyz = PYZ(a.pure)
icon = "icon.ico" if sys.platform == "win32" else "icon.png"
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="plainquant",
    console=False,
    icon=icon,
)
coll = COLLECT(exe, a.binaries, a.datas, name="plainquant")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="plainquant.app",
        icon="icon.png",
        bundle_identifier="io.github.nikshay1.plainquant",
        info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": "0.2.0"},
    )
