"""Convert the official RMUC 2026 STEP assembly into a bounded MuJoCo asset.

The STEP B-rep remains the source of truth for the visual model.  Real-time
contact uses a separately recorded single-valued heightfield proxy: feeding the
complete CAD triangle soup to MuJoCo would turn every screw, railing and
decorative part into a convex collision object.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterable

import numpy as np

from .ramp_audit import audit_fixed_fly_ramps


OFFICIAL_STEP_URL = (
    "https://hz-rm-bbs-web-prod.oss-cn-hangzhou.aliyuncs.com/"
    "f637e13eadfc4602a76c7a05c54f71271782803452461/RMUC2026_V2.0.0.stp"
)
OFFICIAL_STEP_SIZE = 1_254_821_405
OFFICIAL_STEP_SHA256 = "8dfe9ebd761e44d91361b3e593bc05416329112217b58cb35800b3cde2ffae33"
EXPECTED_STEP_PRODUCT = "00_RMUC2026_FINALS_ASM"
OFFICIAL_CORE_BATTLEFIELD_SIZE_M = (28.0, 15.0)
OFFICIAL_RULEBOOK_URL = (
    "https://bbs-web-static.robomaster.com/"
    "5fbd46b110e542faaea519a6dd3065761782460789174/"
    "RoboMaster%202026%20%E6%9C%BA%E7%94%B2%E5%A4%A7%E5%B8%88%E8%B6%85%E7%BA%A7%"
    "E5%AF%B9%E6%8A%97%E8%B5%9B%E6%AF%94%E8%B5%9B%E8%A7%84%E5%88%99%E6%89%8B%"
    "E5%86%8CV2.0.0%EF%BC%8820260626%EF%BC%89.pdf"
)
OFFICIAL_RULEBOOK_SIZE = 21_012_597
OFFICIAL_RULEBOOK_SHA256 = "59d65aac5bac75fd6bbab157e224eb936b49cd2b9d60381fbe196fad635b473a"
OFFICIAL_RULEBOOK_OVERHEAD_PDF_PAGE = 32

# The currently integrated Fudan wheel asset has a 120 mm wheel diameter.  A
# height-field cell should be interpreted relative to that physical feature,
# rather than as an isolated pixel count.  Three samples across one wheel is a
# deliberately modest first spatial-sampling gate; it does not remove the
# topological limitations of a single-valued 2.5D height field.
REFERENCE_WHEEL_DIAMETER_M = 0.120
RECOMMENDED_HEIGHTFIELD_RESOLUTION_M = 0.040
MIN_FINE_INTERACTION_SAMPLES_PER_WHEEL_DIAMETER = 3.0
VALIDATION_HEIGHTFIELD_RESOLUTION_M = 0.020
MIN_VALIDATION_SAMPLES_PER_WHEEL_DIAMETER = 6.0
MINIMUM_HEIGHTFIELD_RESOLUTION_M = 0.010
MAXIMUM_HEIGHTFIELD_SAMPLES = 10_000_000
HEIGHTFIELD_LARGE_ADJACENT_JUMP_M = 0.20


class FieldBuildError(RuntimeError):
    """Raised when source identity or a conversion invariant is violated."""


@dataclass(frozen=True)
class StepIdentity:
    path: Path
    size_bytes: int
    sha256: str
    product: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path_at_build": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "step_product": self.product,
            "official_download_url": OFFICIAL_STEP_URL,
        }


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _step_header(path: Path, *, limit: int = 64 * 1024) -> str:
    with path.open("rb") as handle:
        return handle.read(limit).decode("latin-1", errors="replace")


def validate_official_step(path: Path, *, verify_hash: bool = True) -> StepIdentity:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FieldBuildError(f"官方STEP不是常规文件：{source}")
    size = source.stat().st_size
    if size != OFFICIAL_STEP_SIZE:
        raise FieldBuildError(f"官方STEP大小不匹配：expected={OFFICIAL_STEP_SIZE}, actual={size}")
    header = _step_header(source)
    if "ISO-10303-21" not in header or EXPECTED_STEP_PRODUCT not in header:
        raise FieldBuildError("STEP头不含官方RMUC 2026 Finals装配体身份")
    digest = sha256_file(source) if verify_hash else "NOT_COMPUTED"
    if verify_hash and digest != OFFICIAL_STEP_SHA256:
        raise FieldBuildError(
            f"官方STEP SHA-256不匹配：expected={OFFICIAL_STEP_SHA256}, actual={digest}"
        )
    return StepIdentity(source, size, digest, EXPECTED_STEP_PRODUCT)


def validate_official_rulebook(path: Path, *, verify_hash: bool = True) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FieldBuildError(f"官方V2.0.0规则手册不是常规文件：{source}")
    size = source.stat().st_size
    if size != OFFICIAL_RULEBOOK_SIZE:
        raise FieldBuildError(
            f"官方V2.0.0规则手册大小不匹配：expected={OFFICIAL_RULEBOOK_SIZE}, actual={size}"
        )
    with source.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise FieldBuildError("官方V2.0.0规则手册缺少PDF文件头")
    if verify_hash:
        digest = sha256_file(source)
        if digest != OFFICIAL_RULEBOOK_SHA256:
            raise FieldBuildError(
                "官方V2.0.0规则手册SHA-256不匹配："
                f"expected={OFFICIAL_RULEBOOK_SHA256}, actual={digest}"
            )
    return source


def _crop_overhead_render(image: Any) -> tuple[Any, tuple[int, int, int, int]]:
    """Remove the white PDF-image margin without guessing field dimensions."""

    from PIL import Image

    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.uint8)
    dark = np.mean(pixels, axis=2) < 240.0
    minimum_support = 5
    x_indices = np.flatnonzero(np.count_nonzero(dark, axis=0) >= minimum_support)
    y_indices = np.flatnonzero(np.count_nonzero(dark, axis=1) >= minimum_support)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise FieldBuildError("官方俯视渲染图没有检测到战场边界")
    crop = (
        int(x_indices[0]),
        int(y_indices[0]),
        int(x_indices[-1] + 1),
        int(y_indices[-1] + 1),
    )
    width = crop[2] - crop[0]
    height = crop[3] - crop[1]
    aspect = width / height
    expected_aspect = OFFICIAL_CORE_BATTLEFIELD_SIZE_M[0] / OFFICIAL_CORE_BATTLEFIELD_SIZE_M[1]
    if not 0.98 * expected_aspect <= aspect <= 1.02 * expected_aspect:
        raise FieldBuildError(
            "官方俯视渲染图裁剪比例与28x15米战场不匹配："
            f"crop={crop}, aspect={aspect:.6f}, expected={expected_aspect:.6f}"
        )
    return Image.fromarray(pixels[crop[1] : crop[3], crop[0] : crop[2]]), crop


def _extract_rulebook_surface_guide(
    rulebook: Path,
    output: Path,
    *,
    cad_outer_xy_bounds_m: list[list[float]],
) -> dict[str, Any]:
    from PIL import Image

    pdfimages = shutil.which("pdfimages")
    if pdfimages is None:
        raise FieldBuildError("系统缺少pdfimages，无法提取官方规则手册俯视图")
    with tempfile.TemporaryDirectory(prefix="rmuc2026-rulebook-") as temporary:
        prefix = Path(temporary) / "page32"
        completed = subprocess.run(
            [
                pdfimages,
                "-f",
                str(OFFICIAL_RULEBOOK_OVERHEAD_PDF_PAGE),
                "-l",
                str(OFFICIAL_RULEBOOK_OVERHEAD_PDF_PAGE),
                "-png",
                "-j",
                str(rulebook),
                str(prefix),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise FieldBuildError("pdfimages提取官方俯视图失败：" + completed.stderr.strip())
        candidates: list[tuple[int, Path, tuple[int, int]]] = []
        for candidate in Path(temporary).glob("page32-*"):
            try:
                with Image.open(candidate) as image:
                    width, height = image.size
                    mode = image.mode
            except Exception:
                continue
            aspect = width / max(height, 1)
            if (
                width >= 1000
                and height >= 500
                and 1.7 <= aspect <= 2.0
                and mode in {"RGB", "RGBA", "CMYK"}
            ):
                candidates.append((width * height, candidate, (width, height)))
        if not candidates:
            raise FieldBuildError("规则手册第32页未找到官方战场俯视渲染图")
        _area, source_image, source_size = max(candidates)
        with Image.open(source_image) as image:
            cropped, crop = _crop_overhead_render(image)
        visual_dir = output / "visual"
        visual_dir.mkdir(parents=True, exist_ok=True)
        destination = visual_dir / "official_rulebook_v2_overhead_surface.png"
        cropped.save(destination, format="PNG", optimize=True)
    bounds = np.asarray(cad_outer_xy_bounds_m, dtype=np.float64)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
        raise FieldBuildError("CAD外包矩形不是有限的2x2 XY边界")
    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0.0):
        raise FieldBuildError(f"CAD外包矩形尺寸无效：{extents.tolist()}")
    return {
        "kind": "official_rulebook_overhead_render_surface_guide",
        "file": str(destination.relative_to(output)),
        "sha256": sha256_file(destination),
        "width_px": int(cropped.width),
        "height_px": int(cropped.height),
        "source_pdf_page": OFFICIAL_RULEBOOK_OVERHEAD_PDF_PAGE,
        "source_image_size_px": list(source_size),
        "source_image_crop_ltrb_px": list(crop),
        "world_mapping": {
            "image_left_to_world": "negative_x_red_side",
            "image_right_to_world": "positive_x_blue_side",
            "image_top_to_world": "positive_y",
            "rectangle": "official_cad_assembly_outer_xy_bounds",
            "world_bounds_xy_m": bounds.tolist(),
            "world_size_xy_m": extents.tolist(),
            "calibration": "diagnostic_outer_bounds_fit_not_survey_homography",
        },
        "physics": False,
        "claim_boundary": (
            "diagnostic illustration extracted from the official V2.0.0 rulebook and fitted to "
            "the official CAD assembly outer rectangle; it includes baked shadows, robots and "
            "obstacle appearance, is not a calibrated 28x15 m core-floor texture, and does not "
            "define contact"
        ),
    }


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON file with a private inode, even if ``path`` is a hard link."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cad_outer_xy_bounds_after_translation(manifest: dict[str, Any]) -> list[list[float]]:
    """Recover the exact CAD outer XY rectangle in the final field coordinate frame."""

    dimensions = manifest.get("dimensions")
    if isinstance(dimensions, dict):
        recorded = dimensions.get("cad_assembly_outer_bounds_after_translation_m")
        if isinstance(recorded, list):
            array = np.asarray(recorded, dtype=np.float64)
            if array.shape == (2, 3) and np.isfinite(array).all():
                return array[:, :2].tolist()

    conversion = manifest.get("conversion")
    spawn = manifest.get("recommended_spawn")
    if not isinstance(conversion, dict) or not isinstance(spawn, dict):
        raise FieldBuildError("基础场地产物缺少CAD坐标变换或出生点")
    axis = conversion.get("axis_and_units")
    if not isinstance(axis, dict):
        raise FieldBuildError("基础场地产物缺少CAD轴/单位合同")
    bounds = np.asarray(axis.get("input_bounds"), dtype=np.float64)
    order = np.asarray(axis.get("axis_order_output_xyz"), dtype=np.int64)
    scale = float(axis.get("unit_scale_to_metres", math.nan))
    if (
        bounds.shape != (2, 3)
        or order.shape != (3,)
        or sorted(order.tolist()) != [0, 1, 2]
        or not np.isfinite(bounds).all()
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise FieldBuildError("基础场地产物CAD轴/单位合同损坏")
    transformed = bounds[:, order] * scale
    transformed[:, 0] -= float(spawn["x_before_translation_m"])
    transformed[:, 1] -= float(spawn["y_before_translation_m"])
    return transformed[:, :2].tolist()


def decorate_field_build(
    base_build: Path,
    rulebook_pdf: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create an immutable decorated derivative without rebuilding the 1.25 GB STEP."""

    base = Path(base_build).expanduser().resolve()
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise FieldBuildError(f"基础场地产物缺少manifest：{manifest_path}")
    base_manifest_bytes = manifest_path.read_bytes()
    base_manifest_sha256 = hashlib.sha256(base_manifest_bytes).hexdigest()
    base_manifest_stat = manifest_path.stat()
    manifest = json.loads(base_manifest_bytes.decode("utf-8"))
    source = manifest.get("source")
    if (
        manifest.get("artifact_type") != "rmuc2026_official_field_mujoco_asset"
        or manifest.get("status") != "PASS"
        or not isinstance(source, dict)
        or source.get("sha256") != OFFICIAL_STEP_SHA256
    ):
        raise FieldBuildError("基础场地产物未绑定通过的官方V2.0.0 STEP")
    records = list(manifest.get("visual_meshes", []))
    collision = manifest.get("collision")
    if not records or not isinstance(collision, dict):
        raise FieldBuildError("基础场地产物缺少视觉或碰撞记录")
    records.extend(
        [
            {"file": collision.get("image_file"), "sha256": collision.get("image_sha256")},
            {"file": collision.get("samples_file"), "sha256": collision.get("samples_sha256")},
        ]
    )
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise FieldBuildError("基础场地产物文件记录损坏")
        path = (base / str(record["file"])).resolve()
        if base not in path.parents or not path.is_file():
            raise FieldBuildError(f"基础场地产物文件缺失或越界：{path}")
        expected = record.get("sha256")
        if isinstance(expected, str) and sha256_file(path) != expected:
            raise FieldBuildError(f"基础场地产物哈希不匹配：{path}")
    rulebook = validate_official_rulebook(rulebook_pdf, verify_hash=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FieldBuildError(f"输出目录已存在，拒绝覆盖：{output}")
    shutil.copytree(base, output, copy_function=_link_or_copy)
    cad_outer_xy_bounds = _cad_outer_xy_bounds_after_translation(manifest)
    surface = _extract_rulebook_surface_guide(
        rulebook,
        output,
        cad_outer_xy_bounds_m=cad_outer_xy_bounds,
    )
    manifest["schema_version"] = max(2, int(manifest.get("schema_version", 1)))
    manifest["derived_from_field_build"] = {
        "path_at_build": str(base),
        "manifest_sha256": base_manifest_sha256,
        "asset_copy_mode": "hardlink_when_same_filesystem_else_copy",
        "manifest_copy_mode": "private_atomic_replace",
    }
    manifest["official_rulebook"] = {
        "version": "V2.0.0",
        "publication_date": "2026-06-26",
        "path_at_build": str(rulebook),
        "size_bytes": OFFICIAL_RULEBOOK_SIZE,
        "sha256": OFFICIAL_RULEBOOK_SHA256,
        "official_download_url": OFFICIAL_RULEBOOK_URL,
    }
    manifest["surface_guide"] = surface
    dimensions = manifest.setdefault("dimensions", {})
    if isinstance(dimensions, dict):
        dimensions["official_rulebook_url"] = OFFICIAL_RULEBOOK_URL
    evidence = manifest.setdefault("evidence_boundary", {})
    if isinstance(evidence, dict):
        evidence["official_rulebook_surface_guide"] = True
        # Repacked builds explicitly carry this false before decoration.  Keep
        # the older compatibility field synchronized so one manifest cannot
        # simultaneously claim that the guide is both present and absent.
        evidence["rulebook_surface_guide_present"] = True
        evidence["surface_guide_defines_physics"] = False
    destination_manifest = output / "manifest.json"
    _write_json_atomically(destination_manifest, manifest)
    if manifest_path.read_bytes() != base_manifest_bytes:
        raise FieldBuildError("派生装饰产物意外修改了基础manifest")
    current_base_stat = manifest_path.stat()
    if (
        current_base_stat.st_dev != base_manifest_stat.st_dev
        or current_base_stat.st_ino != base_manifest_stat.st_ino
    ):
        raise FieldBuildError("派生装饰期间基础manifest身份发生变化")
    destination_stat = destination_manifest.stat()
    if (
        destination_stat.st_dev == current_base_stat.st_dev
        and destination_stat.st_ino == current_base_stat.st_ino
    ):
        raise FieldBuildError("派生manifest仍与基础manifest共享inode")
    print("RMUC2026_FIELD_DECORATE=PASS", flush=True)
    return manifest


def _load_xcaf(step_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    started = time.monotonic()
    document = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    # This 1.25 GB assembly carries almost 300k faces.  Import only appearance
    # data required by the renderer; names, GD&T, layers and material metadata
    # make XCAF transfer dramatically slower without changing field pixels.
    reader.SetLayerMode(False)
    reader.SetNameMode(False)
    reader.SetPropsMode(False)
    reader.SetGDTMode(False)
    reader.SetMatMode(False)
    reader.SetSHUOMode(False)
    reader.SetViewMode(False)
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise FieldBuildError(f"OpenCascade读取STEP失败：status={int(status)}")
    roots = int(reader.NbRootsForTransfer())
    if roots != 1:
        raise FieldBuildError(f"官方STEP根装配体数量变化：expected=1, actual={roots}")
    print(f"RMUC2026_FIELD_STEP_READ=PASS; roots={roots}", flush=True)
    print("RMUC2026_FIELD_XCAF_TRANSFER=RUNNING", flush=True)
    if not reader.Transfer(document):
        raise FieldBuildError("OpenCascade无法把STEP装配体转入XCAF文档")
    print("RMUC2026_FIELD_XCAF_TRANSFER=PASS", flush=True)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    shape = shape_tool.GetOneShape()
    if shape.IsNull():
        raise FieldBuildError("OpenCascade得到空场地形状")

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox, True)
    bounds_mm = [float(value) for value in bbox.Get()]
    counts: dict[str, int] = {}
    for kind, label in ((TopAbs_SOLID, "solids"), (TopAbs_FACE, "faces")):
        explorer = TopExp_Explorer(shape, kind)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        counts[label] = count
    audit = {
        "xcaf_roots": roots,
        "bounds_step_units": bounds_mm,
        "extents_step_units": [
            bounds_mm[3] - bounds_mm[0],
            bounds_mm[4] - bounds_mm[1],
            bounds_mm[5] - bounds_mm[2],
        ],
        **counts,
        "transfer_wall_seconds": time.monotonic() - started,
    }
    return document, shape, audit


def _load_geometry_only(step_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load the exact B-rep without expensive per-face appearance metadata."""

    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    started = time.monotonic()
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise FieldBuildError(f"OpenCascade读取STEP失败：status={int(status)}")
    roots = int(reader.NbRootsForTransfer())
    if roots != 1 or not reader.TransferRoots():
        raise FieldBuildError(f"官方STEP几何根转移失败：roots={roots}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise FieldBuildError("OpenCascade得到空场地形状")
    document = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    shape_tool.AddShape(shape, False)

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox, True)
    bounds_mm = [float(value) for value in bbox.Get()]
    counts: dict[str, int] = {}
    for kind, label in ((TopAbs_SOLID, "solids"), (TopAbs_FACE, "faces")):
        explorer = TopExp_Explorer(shape, kind)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        counts[label] = count
    return (
        document,
        shape,
        {
            "xcaf_roots": roots,
            "bounds_step_units": bounds_mm,
            "extents_step_units": [
                bounds_mm[3] - bounds_mm[0],
                bounds_mm[4] - bounds_mm[1],
                bounds_mm[5] - bounds_mm[2],
            ],
            **counts,
            "transfer_wall_seconds": time.monotonic() - started,
            "appearance_metadata_preserved": False,
        },
    )


def audit_step(step_path: Path, *, verify_hash: bool = True) -> dict[str, Any]:
    identity = validate_official_step(step_path, verify_hash=verify_hash)
    _document, _shape, geometry = _load_geometry_only(identity.path)
    return {
        "schema_version": 1,
        "artifact_type": "rmuc2026_official_step_audit",
        "status": "PASS",
        "source": identity.as_dict(),
        "geometry": geometry,
    }


def _dominant_rgba(mesh: Any) -> tuple[int, int, int, int]:
    try:
        rgba_value = getattr(mesh.visual, "main_color", None)
        if rgba_value is None:
            material = getattr(mesh.visual, "material", None)
            rgba_value = getattr(material, "main_color", None)
        if rgba_value is None:
            raise AttributeError("mesh visual has no main color")
        rgba = np.asarray(rgba_value, dtype=np.uint8).reshape(-1)
    except Exception:
        rgba = np.asarray([170, 170, 170, 255], dtype=np.uint8)
    if rgba.size < 4:
        rgba = np.pad(rgba, (0, 4 - rgba.size), constant_values=255)
    return tuple(int(value) for value in rgba[:4])


def _palette_name(rgba: tuple[int, int, int, int]) -> str:
    rgb = np.asarray(rgba[:3], dtype=np.float64) / 255.0
    high = float(np.max(rgb))
    low = float(np.min(rgb))
    spread = high - low
    if spread < 0.12:
        if high < 0.24:
            return "dark"
        if high > 0.82:
            return "light"
        return "gray"
    r, g, b = rgb
    if r > 1.18 * max(g, b):
        return "red" if g < 0.65 * r else "orange"
    if b > 1.16 * max(r, g):
        return "blue"
    if g > 1.14 * max(r, b):
        return "green"
    if r > 0.55 and g > 0.45 and b < 0.45:
        return "yellow"
    return "gray"


PALETTE_RGBA: dict[str, tuple[float, float, float, float]] = {
    "red": (0.72, 0.10, 0.10, 1.0),
    "blue": (0.08, 0.25, 0.72, 1.0),
    "green": (0.12, 0.48, 0.18, 1.0),
    "yellow": (0.88, 0.67, 0.08, 1.0),
    "orange": (0.88, 0.38, 0.07, 1.0),
    "dark": (0.12, 0.14, 0.17, 1.0),
    "gray": (0.52, 0.54, 0.58, 1.0),
    "light": (0.86, 0.87, 0.89, 1.0),
}


def _cad_material_id(rgba: tuple[int, int, int, int]) -> str:
    return "cad_" + "".join(f"{value:02x}" for value in rgba)


def _scene_meshes(scene: Any) -> list[Any]:
    import trimesh

    if isinstance(scene, trimesh.Trimesh):
        return [scene.copy()]
    dumped = scene.dump(concatenate=False)
    meshes = [mesh for mesh in dumped if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        raise FieldBuildError("glTF中没有可用三角网格")
    return meshes


def _axis_contract(meshes: Iterable[Any]) -> dict[str, Any]:
    bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=np.float64)
    low = np.min(bounds[:, 0, :], axis=0)
    high = np.max(bounds[:, 1, :], axis=0)
    raw_extents = high - low
    scale = 0.001 if float(np.max(raw_extents)) > 100.0 else 1.0
    vertical = int(np.argmin(raw_extents))
    horizontal = [index for index in range(3) if index != vertical]
    x_axis = horizontal[int(raw_extents[horizontal[1]] > raw_extents[horizontal[0]])]
    y_axis = next(index for index in horizontal if index != x_axis)
    order = [x_axis, y_axis, vertical]
    output_extents = raw_extents[order] * scale
    if not (
        25.0 <= output_extents[0] <= 35.0
        and 13.0 <= output_extents[1] <= 20.0
        and 2.0 <= output_extents[2] <= 6.0
    ):
        raise FieldBuildError(
            f"自动轴/单位推断未得到RMUC场地量级：extents={output_extents.tolist()}"
        )
    return {
        "input_bounds": [low.tolist(), high.tolist()],
        "input_extents": raw_extents.tolist(),
        "axis_order_output_xyz": order,
        "unit_scale_to_metres": scale,
        "output_extents_before_ground_shift_m": output_extents.tolist(),
    }


def _apply_axis_contract(meshes: Iterable[Any], contract: dict[str, Any]) -> None:
    order = list(contract["axis_order_output_xyz"])
    scale = float(contract["unit_scale_to_metres"])
    for mesh in meshes:
        mesh.vertices = np.asarray(mesh.vertices[:, order] * scale, dtype=np.float64)
        if not mesh.is_winding_consistent:
            mesh.fix_normals(multibody=True)


def _ground_height(meshes: Iterable[Any], *, bin_width_m: float = 0.01) -> float:
    records: list[tuple[np.ndarray, np.ndarray]] = []
    low = math.inf
    high = -math.inf
    for mesh in meshes:
        if len(mesh.faces) == 0:
            continue
        normals = np.asarray(mesh.face_normals)
        mask = normals[:, 2] > 0.85
        if not np.any(mask):
            continue
        heights = np.asarray(mesh.triangles_center[mask, 2], dtype=np.float64)
        areas = np.asarray(mesh.area_faces[mask], dtype=np.float64)
        records.append((heights, areas))
        low = min(low, float(np.min(heights)))
        high = max(high, float(np.max(heights)))
    if not records or not math.isfinite(low) or high <= low:
        raise FieldBuildError("无法从官方场地向上表面识别主地面")
    edges = np.arange(low, high + bin_width_m * 1.5, bin_width_m)
    histogram = np.zeros(len(edges) - 1, dtype=np.float64)
    for heights, areas in records:
        histogram += np.histogram(heights, bins=edges, weights=areas)[0]
    winner = int(np.argmax(histogram))
    center = 0.5 * (edges[winner] + edges[winner + 1])
    selected_heights: list[np.ndarray] = []
    selected_areas: list[np.ndarray] = []
    for heights, areas in records:
        mask = np.abs(heights - center) <= bin_width_m
        if np.any(mask):
            selected_heights.append(heights[mask])
            selected_areas.append(areas[mask])
    values = np.concatenate(selected_heights)
    weights = np.concatenate(selected_areas)
    return float(np.average(values, weights=weights))


def _mesh_set_metrics(meshes: Iterable[Any]) -> dict[str, Any]:
    usable = [mesh for mesh in meshes if len(mesh.faces) > 0]
    if not usable:
        raise FieldBuildError("网格集合没有三角形")
    bounds = np.asarray([mesh.bounds for mesh in usable], dtype=np.float64)
    low = np.min(bounds[:, 0, :], axis=0)
    high = np.max(bounds[:, 1, :], axis=0)
    area = float(sum(float(mesh.area) for mesh in usable))
    faces = int(sum(len(mesh.faces) for mesh in usable))
    if not np.isfinite(low).all() or not np.isfinite(high).all() or not math.isfinite(area):
        raise FieldBuildError("网格集合几何指标含非有限值")
    return {
        "mesh_parts": len(usable),
        "faces": faces,
        "surface_area_m2": area,
        "bounds_m": [low.tolist(), high.tolist()],
        "extents_m": (high - low).tolist(),
    }


def _allocate_part_face_targets(
    face_counts: list[int],
    *,
    budget: int,
    minimum_faces: int = 64,
) -> list[int]:
    if not face_counts:
        return []
    floors = [min(count, minimum_faces) for count in face_counts]
    floor_total = sum(floors)
    if budget < floor_total:
        raise FieldBuildError(
            f"视觉面数预算不足以逐零件保形：available={budget}, minimum_required={floor_total}"
        )
    capacity = [count - floor for count, floor in zip(face_counts, floors)]
    available = min(budget - floor_total, sum(capacity))
    if available <= 0 or sum(capacity) <= 0:
        return floors
    shares = np.asarray(capacity, dtype=np.float64) * (available / sum(capacity))
    additions = np.floor(shares).astype(np.int64)
    remaining = available - int(np.sum(additions))
    order = np.argsort(-(shares - additions), kind="stable")
    for index in order[:remaining]:
        additions[int(index)] += 1
    return [floor + int(extra) for floor, extra in zip(floors, additions)]


def _collision_resolution_contract(
    resolution_m: float,
    *,
    reference_wheel_diameter_m: float = REFERENCE_WHEEL_DIAMETER_M,
) -> dict[str, Any]:
    """Classify height-field sampling relative to a reference wheel diameter."""

    resolution = float(resolution_m)
    wheel_diameter = float(reference_wheel_diameter_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise FieldBuildError("高度图分辨率必须是有限正数")
    if not math.isfinite(wheel_diameter) or wheel_diameter <= 0.0:
        raise FieldBuildError("参考轮径必须是有限正数")
    samples_per_wheel = wheel_diameter / resolution
    fine_sampling = samples_per_wheel + 1.0e-12 >= MIN_FINE_INTERACTION_SAMPLES_PER_WHEEL_DIAMETER
    validation_sampling = samples_per_wheel + 1.0e-12 >= MIN_VALIDATION_SAMPLES_PER_WHEEL_DIAMETER
    return {
        "reference_feature": "Fudan integrated wheel diameter",
        "reference_wheel_diameter_m": wheel_diameter,
        "resolution_m": resolution,
        "samples_per_reference_wheel_diameter": samples_per_wheel,
        "minimum_fine_interaction_samples_per_wheel_diameter": (
            MIN_FINE_INTERACTION_SAMPLES_PER_WHEEL_DIAMETER
        ),
        "recommended_resolution_m": RECOMMENDED_HEIGHTFIELD_RESOLUTION_M,
        "validation_reference_resolution_m": VALIDATION_HEIGHTFIELD_RESOLUTION_M,
        "fine_interaction_spatial_sampling_ready": fine_sampling,
        "validation_spatial_sampling_ready": validation_sampling,
        "classification": (
            "validation_reference_or_better"
            if validation_sampling
            else "fine_reference_or_better"
            if fine_sampling
            else "coarse_compatibility_only"
        ),
        "compatibility_supported": True,
        "recommendation": (
            "spatial sampling meets the six-cells-per-wheel validation reference"
            if validation_sampling
            else "repack at 0.02 m before registering final static validation routes"
            if fine_sampling
            else "repack at 0.04 m before detailed wheel-obstacle interaction work"
        ),
        "claim_boundary": (
            "this gate covers horizontal sampling only; even a passing grid remains a "
            "single-valued 2.5D proxy, not exact CAD collision"
        ),
    }


def _static_route_validation_scope(collision: dict[str, Any]) -> dict[str, Any]:
    """Declare the bounded future policy-evaluation scope of this field asset."""

    quality = collision.get("resolution_quality")
    structural = collision.get("structural_audit")
    if not isinstance(quality, dict) or not isinstance(structural, dict):
        raise FieldBuildError("静态路线验收范围要求完整碰撞分辨率与结构审计")
    spatial_ready = bool(quality.get("validation_spatial_sampling_ready", False))
    topology_supported = bool(structural.get("underpasses_and_overhangs_preserved", False))
    blockers = [
        "official_core_landmark_coordinate_calibration_not_yet_frozen",
        "driveable_static_route_registry_not_yet_frozen",
        "collision_candidate_semantics_not_yet_manually_accepted",
        "field_material_zones_and_friction_robustness_not_yet_registered",
        "dynamic_facilities_not_modeled_as_dynamic_bodies",
    ]
    if not spatial_ready:
        blockers.insert(0, "heightfield_below_six_samples_per_120mm_wheel")
    if not topology_supported:
        blockers.insert(0, "single_valued_heightfield_cannot_preserve_multilevel_topology")
    return {
        "profile_id": "RMUC2026_STATIC_ROUTE_V1_DRAFT",
        "status": "DRAFT_BLOCKED",
        "final_policy_validation_ready": False,
        "spatial_sampling_ready": spatial_ready,
        "whole_field_topology_ready": topology_supported,
        "eligible_future_scope": (
            "pre-registered static, single-valued drive corridors after landmark, "
            "semantic, material and route acceptance"
        ),
        "eligible_surface_types": [
            "flat_recovery_zone",
            "single_valued_floor",
            "manually_accepted_ramp_or_step_top_surface",
        ],
        "excluded_until_modeled": [
            "underpass",
            "overhang",
            "stacked_or_multilevel_surface",
            "moving_or_actuated_field_element",
            "unverified_thin_wall_or_collision_candidate",
        ],
        "required_policy_scenarios": [
            "upright_stand",
            "left_right_front_rear_fall_recovery",
            "forward_reverse_tracking",
            "left_right_turning",
            "ramp_and_step_ascent_descent",
            "jump_landing_and_recovery",
        ],
        "fall_recovery_gate": {
            "recover_within_s": 5.0,
            "upright_tilt_max_deg": 15.0,
            "stable_hold_s": 2.0,
            "horizontal_drift_max_m": 0.5,
            "external_reset_or_pose_teleport_allowed": False,
            "capability_belongs_to_policy_not_field": True,
        },
        "blocking_reasons": blockers,
        "claim_boundary": (
            "a PASS field build proves asset integrity and bounded static interaction only; "
            "it does not prove whole-field topology, robot recovery, policy quality, or "
            "source-to-target equivalence"
        ),
    }


def _heightfield_structural_audit(
    height_m: np.ndarray,
    *,
    large_jump_threshold_m: float = HEIGHTFIELD_LARGE_ADJACENT_JUMP_M,
) -> dict[str, Any]:
    """Measure grid discontinuities and state the unavoidable 2.5D topology boundary."""

    height = np.asarray(height_m, dtype=np.float64)
    threshold = float(large_jump_threshold_m)
    if height.ndim != 2 or min(height.shape) < 2:
        raise FieldBuildError("高度图结构审计要求至少2x2的二维数组")
    if not np.isfinite(height).all():
        raise FieldBuildError("高度图结构审计输入含非有限值")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise FieldBuildError("高度跳变阈值必须是有限正数")

    jump_x = np.abs(np.diff(height, axis=1))
    jump_y = np.abs(np.diff(height, axis=0))
    large_x = int(np.count_nonzero(jump_x > threshold))
    large_y = int(np.count_nonzero(jump_y > threshold))
    max_x = float(np.max(jump_x))
    max_y = float(np.max(jump_y))
    return {
        "large_jump_threshold_m": threshold,
        "adjacent_edges": {
            "x": int(jump_x.size),
            "y": int(jump_y.size),
            "total": int(jump_x.size + jump_y.size),
        },
        "large_jump_edges": {
            "x": large_x,
            "y": large_y,
            "total": large_x + large_y,
        },
        "maximum_adjacent_height_jump_m": {
            "x": max_x,
            "y": max_y,
            "overall": max(max_x, max_y),
        },
        "height_range_m": [float(np.min(height)), float(np.max(height))],
        "representation": "single_height_per_xy_cell_2.5D",
        "ray_selection_rule": "highest downward ray hit at each xy sample",
        "underpasses_and_overhangs_preserved": False,
        "sealed_underpass_risk": True,
        "exact_brep_collision": False,
        "structural_declaration": (
            "a highest-surface height field seals underpasses and cannot represent stacked, "
            "vertical, or overhanging collision surfaces"
        ),
    }


def _mesh_watertight_if_available(mesh: Any) -> bool | None:
    try:
        return bool(mesh.is_watertight)
    except Exception:
        return None


def _clean_visual_mesh(
    mesh: Any,
    *,
    vertex_digits: int = 8,
    minimum_face_area_m2: float = 1.0e-14,
) -> dict[str, Any]:
    """Remove renderer-hostile duplicate data without changing visible geometry.

    Official CAD exports repeat many vertices and sometimes repeat the same
    triangle with opposite winding.  MuJoCo's shadow mapping is particularly
    sensitive to those coincident faces.  The tolerance here is ten nanometres
    in the axis-normalized metre frame, far below the source and heightfield
    precision, so this cleanup cannot act as geometric simplification.
    """

    input_vertices = int(len(mesh.vertices))
    input_faces = int(len(mesh.faces))
    input_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    input_area = float(mesh.area)
    if input_faces <= 0 or not np.isfinite(input_bounds).all() or not math.isfinite(input_area):
        raise FieldBuildError("视觉网格清理要求有限且非空的输入几何")

    mesh.merge_vertices(digits_vertex=vertex_digits)
    after_vertex_merge = int(len(mesh.vertices))

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    valid_faces = np.isfinite(face_areas) & (face_areas > minimum_face_area_m2)
    invalid_faces_removed = int(len(valid_faces) - np.count_nonzero(valid_faces))
    if invalid_faces_removed:
        mesh.update_faces(valid_faces)

    unique_faces = np.asarray(mesh.unique_faces(), dtype=bool)
    duplicate_faces_removed = int(len(unique_faces) - np.count_nonzero(unique_faces))
    if duplicate_faces_removed:
        mesh.update_faces(unique_faces)
    mesh.remove_unreferenced_vertices()

    winding_consistent_before_repair = bool(mesh.is_winding_consistent)
    if not winding_consistent_before_repair:
        mesh.fix_normals(multibody=True)
    winding_consistent_after_repair = bool(mesh.is_winding_consistent)

    output_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    output_area = float(mesh.area)
    output_face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if (
        len(mesh.faces) <= 0
        or not np.isfinite(mesh.vertices).all()
        or not np.isfinite(output_bounds).all()
        or not math.isfinite(output_area)
        or not np.isfinite(output_face_areas).all()
        or np.any(output_face_areas <= minimum_face_area_m2)
        or not np.all(mesh.unique_faces())
    ):
        raise FieldBuildError("视觉网格清理产生无效几何")

    return {
        "algorithm": "merge_10nm_vertices_remove_degenerate_and_duplicate_faces_repair_winding",
        "vertex_decimal_digits_m": vertex_digits,
        "minimum_face_area_m2": minimum_face_area_m2,
        "input_vertices": input_vertices,
        "output_vertices": int(len(mesh.vertices)),
        "merged_vertices": input_vertices - after_vertex_merge,
        "unreferenced_vertices_removed": after_vertex_merge - int(len(mesh.vertices)),
        "input_faces": input_faces,
        "output_faces": int(len(mesh.faces)),
        "invalid_faces_removed": invalid_faces_removed,
        "duplicate_faces_removed": duplicate_faces_removed,
        "input_surface_area_m2": input_area,
        "output_surface_area_m2": output_area,
        "bounds_max_abs_change_m": float(np.max(np.abs(output_bounds - input_bounds))),
        "winding_consistent_before_repair": winding_consistent_before_repair,
        "winding_consistent_after_repair": winding_consistent_after_repair,
    }


def _simplify_parts_then_group(
    meshes: list[Any],
    output: Path,
    *,
    target_faces: int,
    preserve_part_faces: int = 5_000,
    preserve_broad_xy_area_m2: float = 25.0,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
    """Simplify each CAD part independently, then concatenate only for draw calls.

    Merging all same-colour parts before decimation destroys disconnected broad plates: a
    230-face, 889 m² floor competes with millions of tiny same-colour triangles.  This path
    protects every small part and every broad XY-spanning part before any colour grouping.
    """

    import trimesh

    candidates: list[dict[str, Any]] = []
    for index, mesh in enumerate(meshes):
        if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
            continue
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        xy_area = float(np.prod(np.maximum(bounds[1, :2] - bounds[0, :2], 0.0)))
        reasons: list[str] = []
        if len(mesh.faces) <= preserve_part_faces:
            reasons.append("small_part_face_threshold")
        if xy_area >= preserve_broad_xy_area_m2:
            reasons.append("broad_xy_footprint")
        candidates.append(
            {
                "index": index,
                "mesh": mesh,
                "rgba8": _dominant_rgba(mesh),
                "input_vertices": int(len(mesh.vertices)),
                "input_faces": int(len(mesh.faces)),
                "input_area_m2": float(mesh.area),
                "input_bounds_m": bounds.tolist(),
                "input_is_watertight": _mesh_watertight_if_available(mesh),
                "xy_bbox_area_m2": xy_area,
                "protection_reasons": reasons,
            }
        )
    input_faces = sum(int(row["input_faces"]) for row in candidates)
    if input_faces <= 0:
        raise FieldBuildError("官方场地网格没有有限三角形")
    raw_candidate_metrics = _mesh_set_metrics([row["mesh"] for row in candidates])
    raw_candidate_extents = np.asarray(raw_candidate_metrics["extents_m"], dtype=np.float64)
    outer_xy_area = float(raw_candidate_extents[0] * raw_candidate_extents[1])
    base_surface_candidates = [
        row
        for row in candidates
        if float(row["xy_bbox_area_m2"]) >= 0.90 * outer_xy_area
        and float(row["input_area_m2"]) >= 0.25 * float(raw_candidate_metrics["surface_area_m2"])
    ]
    if len(base_surface_candidates) > 1:
        base_surface_candidates = [
            max(base_surface_candidates, key=lambda row: float(row["input_area_m2"]))
        ]
    base_surface_indices = {int(row["index"]) for row in base_surface_candidates}
    for row in candidates:
        row["visual_role"] = (
            "base_surface_shell" if int(row["index"]) in base_surface_indices else "cad_structure"
        )

    protected = [row for row in candidates if row["protection_reasons"]]
    reducible = [row for row in candidates if not row["protection_reasons"]]
    protected_faces = sum(int(row["input_faces"]) for row in protected)
    targets = _allocate_part_face_targets(
        [int(row["input_faces"]) for row in reducible],
        budget=max(0, target_faces - protected_faces),
    )
    for row, wanted in zip(reducible, targets):
        row["target_faces"] = wanted
    for row in protected:
        row["target_faces"] = int(row["input_faces"])

    simplified_by_role_and_rgba: dict[tuple[str, tuple[int, int, int, int]], list[Any]] = (
        defaultdict(list)
    )
    part_records: list[dict[str, Any]] = []
    for row in candidates:
        original = row["mesh"]
        # Geometry is exported with one material per grouped OBJ, so copying the
        # source visual here is both unnecessary and may trigger optional SciPy
        # colour conversions inside trimesh.
        working = trimesh.Trimesh(
            vertices=np.asarray(original.vertices, dtype=np.float64).copy(),
            faces=np.asarray(original.faces, dtype=np.int64).copy(),
            process=False,
        )
        cleanup_before_simplification = _clean_visual_mesh(working)
        wanted = int(row["target_faces"])
        if len(working.faces) > wanted:
            try:
                working = working.simplify_quadric_decimation(face_count=wanted)
            except Exception as exc:
                raise FieldBuildError(
                    "逐零件简化失败："
                    f"part={row['index']}, faces={len(original.faces)}, "
                    f"target={wanted}, error={type(exc).__name__}: {exc}"
                ) from exc
        cleanup_after_simplification = _clean_visual_mesh(working)
        if (
            len(working.faces) <= 0
            or len(working.faces) > len(original.faces)
            or not np.isfinite(working.vertices).all()
            or not math.isfinite(float(working.area))
        ):
            raise FieldBuildError(f"逐零件简化产生无效几何：part={row['index']}")
        rgba8 = row["rgba8"]
        visual_role = str(row["visual_role"])
        simplified_by_role_and_rgba[(visual_role, rgba8)].append(working)
        input_bounds = np.asarray(row["input_bounds_m"], dtype=np.float64)
        output_bounds = np.asarray(working.bounds, dtype=np.float64)
        part_records.append(
            {
                "source_part_index": int(row["index"]),
                "material_id": _cad_material_id(rgba8),
                "visual_role": visual_role,
                "input_vertices": int(row["input_vertices"]),
                "output_vertices": int(len(working.vertices)),
                "input_faces": int(row["input_faces"]),
                "cleaned_input_faces": int(cleanup_before_simplification["output_faces"]),
                "target_faces": wanted,
                "output_faces": int(len(working.faces)),
                "input_surface_area_m2": float(row["input_area_m2"]),
                "cleaned_input_surface_area_m2": float(
                    cleanup_before_simplification["output_surface_area_m2"]
                ),
                "output_surface_area_m2": float(working.area),
                "input_bounds_m": input_bounds.tolist(),
                "output_bounds_m": output_bounds.tolist(),
                "input_height_range_m": input_bounds[:, 2].tolist(),
                "output_height_range_m": output_bounds[:, 2].tolist(),
                "input_height_extent_m": float(input_bounds[1, 2] - input_bounds[0, 2]),
                "output_height_extent_m": float(output_bounds[1, 2] - output_bounds[0, 2]),
                "input_is_watertight": row["input_is_watertight"],
                "output_is_watertight": _mesh_watertight_if_available(working),
                "xy_bbox_area_m2": float(row["xy_bbox_area_m2"]),
                "protection_reasons": list(row["protection_reasons"]),
                "visual_cleanup": {
                    "before_simplification": cleanup_before_simplification,
                    "after_simplification": cleanup_after_simplification,
                },
            }
        )

    visual_dir = output / "visual"
    visual_dir.mkdir()
    records: list[dict[str, Any]] = []
    final_meshes: list[Any] = []
    for visual_role, rgba8 in sorted(simplified_by_role_and_rgba):
        base_material_id = _cad_material_id(rgba8)
        material_id = (
            f"cad_base_surface_{base_material_id.removeprefix('cad_')}"
            if visual_role == "base_surface_shell"
            else base_material_id
        )
        source_rows = [
            row for row in candidates if row["rgba8"] == rgba8 and row["visual_role"] == visual_role
        ]
        merged = trimesh.util.concatenate(simplified_by_role_and_rgba[(visual_role, rgba8)])
        group_cleanup = _clean_visual_mesh(merged)
        path = visual_dir / f"rmuc2026_{material_id}.obj"
        merged.export(path, file_type="obj", include_color=False)
        final_meshes.append(merged)
        record = {
            "material_id": material_id,
            "visual_role": visual_role,
            "source_rgba8": list(rgba8),
            "rgba": [float(value) / 255.0 for value in rgba8],
            "file": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "source_parts": len(source_rows),
            "source_part_indices": sorted(int(row["index"]) for row in source_rows),
            "protected_source_parts": sum(bool(row["protection_reasons"]) for row in source_rows),
            "input_faces": sum(int(row["input_faces"]) for row in source_rows),
            "output_faces": int(len(merged.faces)),
            "vertices": int(len(merged.vertices)),
            "input_surface_area_m2": sum(float(row["input_area_m2"]) for row in source_rows),
            "output_surface_area_m2": float(merged.area),
            "visual_cleanup": group_cleanup,
        }
        records.append(record)

    raw_metrics = _mesh_set_metrics([row["mesh"] for row in candidates])
    output_metrics = _mesh_set_metrics(final_meshes)
    raw_bounds = np.asarray(raw_metrics["bounds_m"], dtype=np.float64)
    output_bounds = np.asarray(output_metrics["bounds_m"], dtype=np.float64)
    raw_extents = raw_bounds[1] - raw_bounds[0]
    output_extents = output_bounds[1] - output_bounds[0]
    area_retention = float(output_metrics["surface_area_m2"] / raw_metrics["surface_area_m2"])
    protected_exact = all(
        row["cleaned_input_faces"] == row["output_faces"]
        and math.isclose(
            float(row["cleaned_input_surface_area_m2"]),
            float(row["output_surface_area_m2"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        for row in part_records
        if row["protection_reasons"]
    )
    bounds_max_abs_error = float(np.max(np.abs(output_bounds - raw_bounds)))
    gates = {
        "protected_parts_exact": protected_exact,
        "surface_area_retention_at_least_0_90": area_retention >= 0.90,
        "bounds_max_abs_error_at_most_0_03_m": bounds_max_abs_error <= 0.03,
        "each_extent_retained_within_1_percent": bool(
            np.all(output_extents >= 0.99 * raw_extents)
            and np.all(output_extents <= 1.01 * raw_extents)
        ),
        "face_budget_not_exceeded": int(output_metrics["faces"])
        <= max(target_faces, protected_faces),
        "visual_cleanup_completed_for_every_part": all(
            set(row["visual_cleanup"]) == {"before_simplification", "after_simplification"}
            for row in part_records
        ),
        "final_groups_have_unique_finite_faces": all(
            int(record["visual_cleanup"]["output_faces"]) > 0
            and int(record["visual_cleanup"]["output_vertices"]) > 0
            for record in records
        ),
    }
    if not all(gates.values()):
        raise FieldBuildError(f"逐零件视觉保真硬门失败：{gates}")
    source_part_indices = [int(row["source_part_index"]) for row in part_records]
    grouped_part_indices = [
        int(index) for record in records for index in record["source_part_indices"]
    ]
    geometry_gates = {
        "source_part_indices_unique": len(source_part_indices) == len(set(source_part_indices)),
        "group_membership_exact": sorted(source_part_indices) == sorted(grouped_part_indices),
        "input_and_output_bounds_finite": all(
            np.isfinite(np.asarray(row["input_bounds_m"], dtype=np.float64)).all()
            and np.isfinite(np.asarray(row["output_bounds_m"], dtype=np.float64)).all()
            for row in part_records
        ),
        "input_and_output_counts_positive": all(
            int(row["input_vertices"]) > 0
            and int(row["output_vertices"]) > 0
            and int(row["input_faces"]) > 0
            and int(row["output_faces"]) > 0
            for row in part_records
        ),
    }
    if not all(geometry_gates.values()):
        raise FieldBuildError(f"逐零件空间追踪硬门失败：{geometry_gates}")
    audit = {
        "algorithm": "clean_then_simplify_each_original_part_then_clean_group_by_exact_rgba",
        "visual_cleanup": {
            "purpose": "remove coincident/degenerate faces that cause MuJoCo render artifacts",
            "geometry_simplification": False,
            "collision_changed": False,
            "vertex_decimal_digits_m": 8,
            "minimum_face_area_m2": 1.0e-14,
            "part_invalid_faces_removed": sum(
                int(row["visual_cleanup"]["before_simplification"]["invalid_faces_removed"])
                + int(row["visual_cleanup"]["after_simplification"]["invalid_faces_removed"])
                for row in part_records
            ),
            "part_duplicate_faces_removed": sum(
                int(row["visual_cleanup"]["before_simplification"]["duplicate_faces_removed"])
                + int(row["visual_cleanup"]["after_simplification"]["duplicate_faces_removed"])
                for row in part_records
            ),
            "group_invalid_faces_removed": sum(
                int(record["visual_cleanup"]["invalid_faces_removed"]) for record in records
            ),
            "group_duplicate_faces_removed": sum(
                int(record["visual_cleanup"]["duplicate_faces_removed"]) for record in records
            ),
        },
        "target_visual_faces": target_faces,
        "preserve_part_faces_at_most": preserve_part_faces,
        "preserve_broad_xy_bbox_area_at_least_m2": preserve_broad_xy_area_m2,
        "source_parts": len(candidates),
        "protected_parts": len(protected),
        "base_surface_parts": sorted(base_surface_indices),
        "protected_faces": protected_faces,
        "raw_geometry": raw_metrics,
        "output_geometry": output_metrics,
        "surface_area_retention": area_retention,
        "bounds_max_abs_error_m": bounds_max_abs_error,
        "hard_gates": gates,
        "part_geometry_audit": {
            "schema_version": 1,
            "record_location": "conversion.visual_simplification.parts",
            "record_count": len(part_records),
            "source_coordinate_frame": (
                "axis-normalized metres with main floor shifted to z=0, before spawn translation"
            ),
            "final_world_translation_m": None,
            "final_world_bounds_formula": (
                "part input/output bounds plus final_world_translation_m"
            ),
            "group_membership_location": "visual_meshes[*].source_part_indices",
            "watertight_value_semantics": "boolean when trimesh can evaluate it, otherwise null",
            "hard_gates": geometry_gates,
        },
        "parts": part_records,
    }
    return records, final_meshes, audit


def _simplify_groups(
    meshes: list[Any], output: Path, *, target_faces: int
) -> tuple[list[dict[str, Any]], list[Any]]:
    records, final_meshes, _audit = _simplify_parts_then_group(
        meshes, output, target_faces=target_faces
    )
    return records, final_meshes


def _ray_heightfield(
    meshes: list[Any],
    output: Path,
    *,
    resolution_m: float,
    source_glb_sha256: str | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    import trimesh
    from PIL import Image

    combined = trimesh.util.concatenate(meshes)
    low, high = np.asarray(combined.bounds, dtype=np.float64)
    x, y = _heightfield_grid_axes(low, high, resolution_m=resolution_m)
    columns = len(x)
    sample_count = len(y) * columns
    heights = np.full(sample_count, np.nan, dtype=np.float64)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(combined)
    batch = 8192
    for start in range(0, sample_count, batch):
        stop = min(start + batch, sample_count)
        flat_indices = np.arange(start, stop, dtype=np.int64)
        rows = flat_indices // columns
        column_indices = flat_indices % columns
        origins = np.column_stack(
            (x[column_indices], y[rows], np.full(len(flat_indices), high[2] + 1.0))
        )
        directions = np.zeros_like(origins)
        directions[:, 2] = -1.0
        locations, ray_indices, _triangle_indices = intersector.intersects_location(
            origins,
            directions,
            multiple_hits=False,
        )
        if len(ray_indices):
            heights[start + np.asarray(ray_indices, dtype=np.int64)] = locations[:, 2]
    missing = ~np.isfinite(heights)
    height = heights.reshape((len(y), len(x)))
    height[~np.isfinite(height)] = 0.0
    height = np.maximum(height, 0.0)
    # Suppress isolated decorative spikes while retaining ramps and broad platforms.
    padded = np.pad(height, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    median = np.median(windows, axis=(-1, -2))
    isolated = np.abs(height - median) > 0.20
    height[isolated] = median[isolated]
    from .wall_tip_repair import SOURCE_GLB_SHA256, repair_verified_wall_tips

    if source_glb_sha256 == SOURCE_GLB_SHA256 and resolution_m == 0.01:
        try:
            wall_tip_repair = repair_verified_wall_tips(
                meshes, x, y, height, source_glb_sha256=source_glb_sha256
            )
        except ValueError as exc:
            raise FieldBuildError(f"官方墙端碰撞采样修复失败：{exc}") from exc
    else:
        wall_tip_repair = {
            "status": "NOT_APPLICABLE",
            "reason": "wall-end repair is audited only for the exact official GLB on a 1 cm grid",
            "source_glb_sha256": source_glb_sha256,
            "grid_resolution_m": resolution_m,
            "repaired_nodes": 0,
        }
    max_height = float(np.max(height))
    if not math.isfinite(max_height) or max_height <= 0.01:
        raise FieldBuildError("官方场地高度图没有检测到有效障碍")
    normalized = np.clip(height / max_height, 0.0, 1.0)
    pixels = np.rint(normalized * 65535.0).astype(np.uint16)
    collision_dir = output / "collision"
    collision_dir.mkdir()
    image_path = collision_dir / "rmuc2026_heightfield.png"
    # MuJoCo's PNG hfield loader maps the first image row to positive local Y,
    # while our NumPy grid is indexed from y_min to y_max.  Flip rows here so
    # the collision samples and STEP visual occupy the same world locations.
    Image.fromarray(np.flipud(pixels), mode="I;16").save(image_path)
    npz_path = collision_dir / "rmuc2026_heightfield.npz"
    np.savez_compressed(npz_path, x_m=x, y_m=y, height_m=height)
    wall_tip_repair["collision_samples_sha256"] = sha256_file(npz_path)
    record = {
        "kind": "top_surface_heightfield_proxy",
        "image_file": str(image_path.relative_to(output)),
        "image_sha256": sha256_file(image_path),
        "samples_file": str(npz_path.relative_to(output)),
        "samples_sha256": sha256_file(npz_path),
        "resolution_m": resolution_m,
        "rows_y": int(len(y)),
        "columns_x": int(len(x)),
        "x_min_m": float(x[0]),
        "x_max_m": float(x[-1]),
        "y_min_m": float(y[0]),
        "y_max_m": float(y[-1]),
        "maximum_height_m": max_height,
        "minimum_height_m": float(np.min(height)),
        "ray_misses_filled_with_ground": int(np.count_nonzero(missing)),
        "isolated_spikes_replaced": int(np.count_nonzero(isolated)),
        "png_rows": "flipped_y_for_mujoco_hfield_loader",
        "resolution_quality": _collision_resolution_contract(resolution_m),
        "structural_audit": _heightfield_structural_audit(height),
        "claim_boundary": (
            "official STEP-derived single-valued 2.5D top-surface contact proxy; not exact "
            "B-rep contact; ray misses are filled with ground and no validity mask is saved; "
            "highest-surface sampling seals underpasses and cannot preserve stacked, vertical, "
            "or overhanging collision surfaces"
        ),
    }
    record["verified_wall_tip_repair"] = wall_tip_repair
    return record, x, y, height


def _heightfield_grid_axes(
    low_xyz_m: np.ndarray,
    high_xyz_m: np.ndarray,
    *,
    resolution_m: float,
    maximum_samples: int = MAXIMUM_HEIGHTFIELD_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate bounded XY axes before creating the much larger ray grid.

    A 1 cm field is useful for seam evaluation but is roughly four times the
    sample count of a 2 cm field.  Apply the same ten-million-sample ceiling as
    the runtime-pack validator before ``meshgrid`` and ray-origin allocation so
    malformed bounds cannot exhaust memory during a local build.
    """

    low = np.asarray(low_xyz_m, dtype=np.float64)
    high = np.asarray(high_xyz_m, dtype=np.float64)
    resolution = float(resolution_m)
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(high <= low)
    ):
        raise FieldBuildError("高度图边界必须是有限且递增的三维范围")
    if not MINIMUM_HEIGHTFIELD_RESOLUTION_M <= resolution <= 0.25:
        raise FieldBuildError(
            f"高度图分辨率必须在[{MINIMUM_HEIGHTFIELD_RESOLUTION_M:.02f}, 0.25]米"
        )
    if isinstance(maximum_samples, bool) or maximum_samples < 4:
        raise FieldBuildError("高度图采样上限必须是至少4的整数")

    # Keep fine grids on the established 2 cm collision envelope.  Otherwise
    # changing only the sample spacing also shifts the hfield centre, spawn and
    # every translated visual mesh, which breaks pack-to-pack reproducibility.
    # Coarser compatibility grids retain their historical one-cell margin.
    fine_ratio = VALIDATION_HEIGHTFIELD_RESOLUTION_M / resolution
    fine_stride = int(round(fine_ratio))
    if resolution < VALIDATION_HEIGHTFIELD_RESOLUTION_M and math.isclose(
        fine_ratio, fine_stride, rel_tol=0.0, abs_tol=1.0e-9
    ):
        envelope_step = VALIDATION_HEIGHTFIELD_RESOLUTION_M
        coarse_x = np.arange(
            low[0] - envelope_step,
            high[0] + envelope_step + envelope_step * 0.5,
            envelope_step,
        )
        coarse_y = np.arange(
            low[1] - envelope_step,
            high[1] + envelope_step + envelope_step * 0.5,
            envelope_step,
        )
        x = _subdivide_grid_axis(coarse_x, stride=fine_stride)
        y = _subdivide_grid_axis(coarse_y, stride=fine_stride)
    else:
        margin = resolution
        x = np.arange(low[0] - margin, high[0] + margin + resolution * 0.5, resolution)
        y = np.arange(low[1] - margin, high[1] + margin + resolution * 0.5, resolution)
    samples = int(len(x)) * int(len(y))
    if samples > maximum_samples:
        raise FieldBuildError(
            "高度图网格超过采样上限："
            f"shape={len(y)}x{len(x)}, samples={samples}, limit={maximum_samples}"
        )
    return x, y


def _subdivide_grid_axis(coarse_axis: np.ndarray, *, stride: int) -> np.ndarray:
    """Subdivide an axis while preserving every established coarse coordinate."""

    coarse = np.asarray(coarse_axis, dtype=np.float64)
    fine = np.empty((len(coarse) - 1) * stride + 1, dtype=np.float64)
    fine[::stride] = coarse
    step = float(coarse[1] - coarse[0]) / stride
    for offset in range(1, stride):
        fine[offset::stride] = coarse[:-1] + offset * step
    return fine


def _choose_spawn(x: np.ndarray, y: np.ndarray, height: np.ndarray) -> dict[str, float]:
    resolution = float(x[1] - x[0])
    ratio = VALIDATION_HEIGHTFIELD_RESOLUTION_M / resolution
    stride = int(round(ratio))
    if resolution < VALIDATION_HEIGHTFIELD_RESOLUTION_M and math.isclose(
        ratio, stride, rel_tol=0.0, abs_tol=1.0e-7
    ):
        selection_x = x[::stride]
        selection_y = y[::stride]
        selection_height = height[::stride, ::stride]
    else:
        selection_x = x
        selection_y = y
        selection_height = height
    selection_resolution = float(selection_x[1] - selection_x[0])
    radius_cells = max(2, int(round(0.65 / selection_resolution)))
    local_range = _square_local_range(selection_height, radius_cells=radius_cells)
    margin = 1.2
    candidates = (
        (np.abs(selection_height) <= 0.03)
        & (local_range <= 0.025)
        & (selection_x[None, :] >= selection_x[0] + margin)
        & (selection_x[None, :] <= selection_x[-1] - margin)
        & (selection_y[:, None] >= selection_y[0] + margin)
        & (selection_y[:, None] <= selection_y[-1] - margin)
    )
    if not np.any(candidates):
        raise FieldBuildError("没有找到足够大的官方场地平坦出生区")
    target_x = float(selection_x[0] + 0.78 * (selection_x[-1] - selection_x[0]))
    target_y = float(0.5 * (selection_y[0] + selection_y[-1]))
    score = (selection_x[None, :] - target_x) ** 2 + (selection_y[:, None] - target_y) ** 2
    score[~candidates] = math.inf
    row, column = np.unravel_index(int(np.argmin(score)), score.shape)
    return {
        "x_before_translation_m": float(selection_x[column]),
        "y_before_translation_m": float(selection_y[row]),
        "terrain_height_m": float(selection_height[row, column]),
        "verified_flat_radius_m": radius_cells * selection_resolution,
    }


def _square_local_range(height_m: np.ndarray, *, radius_cells: int) -> np.ndarray:
    """Return edge-padded square-neighbourhood ranges in linear memory."""

    height = np.asarray(height_m, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 1 or not np.isfinite(height).all():
        raise FieldBuildError("出生点局部范围要求有限二维高度图")
    if isinstance(radius_cells, bool) or not isinstance(radius_cells, int) or radius_cells < 0:
        raise FieldBuildError("出生点局部范围半径必须是非负整数")
    if radius_cells == 0:
        return np.zeros_like(height)
    window = radius_cells * 2 + 1
    horizontal_low = np.empty_like(height)
    horizontal_high = np.empty_like(height)
    padded_x = np.pad(height, ((0, 0), (radius_cells, radius_cells)), mode="edge")
    for row in range(height.shape[0]):
        low, high = _sliding_min_max_1d(padded_x[row], window)
        horizontal_low[row] = low
        horizontal_high[row] = high

    local_low = np.empty_like(height)
    local_high = np.empty_like(height)
    padded_low = np.pad(horizontal_low, ((radius_cells, radius_cells), (0, 0)), mode="edge")
    padded_high = np.pad(horizontal_high, ((radius_cells, radius_cells), (0, 0)), mode="edge")
    for column in range(height.shape[1]):
        local_low[:, column] = _sliding_extreme_1d(padded_low[:, column], window, maximum=False)
        local_high[:, column] = _sliding_extreme_1d(padded_high[:, column], window, maximum=True)
    return local_high - local_low


def _sliding_min_max_1d(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=np.float64)
    low = np.empty(len(source) - window + 1, dtype=np.float64)
    high = np.empty_like(low)
    low_indices: deque[int] = deque()
    high_indices: deque[int] = deque()
    for index, value in enumerate(source):
        while low_indices and source[low_indices[-1]] >= value:
            low_indices.pop()
        while high_indices and source[high_indices[-1]] <= value:
            high_indices.pop()
        low_indices.append(index)
        high_indices.append(index)
        expired = index - window
        if low_indices[0] <= expired:
            low_indices.popleft()
        if high_indices[0] <= expired:
            high_indices.popleft()
        if index >= window - 1:
            output_index = index - window + 1
            low[output_index] = source[low_indices[0]]
            high[output_index] = source[high_indices[0]]
    return low, high


def _sliding_extreme_1d(values: np.ndarray, window: int, *, maximum: bool) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.empty(len(source) - window + 1, dtype=np.float64)
    indices: deque[int] = deque()
    for index, value in enumerate(source):
        while indices and (
            source[indices[-1]] <= value if maximum else source[indices[-1]] >= value
        ):
            indices.pop()
        indices.append(index)
        expired = index - window
        if indices[0] <= expired:
            indices.popleft()
        if index >= window - 1:
            output[index - window + 1] = source[indices[0]]
    return output


def _translate_final_assets(
    meshes: list[Any],
    records: list[dict[str, Any]],
    output: Path,
    *,
    spawn: dict[str, float],
) -> None:
    translation = np.asarray(
        [
            -float(spawn["x_before_translation_m"]),
            -float(spawn["y_before_translation_m"]),
            -float(spawn["terrain_height_m"]),
        ],
        dtype=np.float64,
    )
    for mesh, record in zip(meshes, records, strict=True):
        mesh.apply_translation(translation)
        path = output / str(record["file"])
        mesh.export(path, file_type="obj", include_color=False)
        record["sha256"] = sha256_file(path)
        record["bounds_m"] = np.asarray(mesh.bounds, dtype=np.float64).tolist()


def _record_part_geometry_world_bounds(
    simplification: dict[str, Any],
    *,
    spawn: dict[str, float],
) -> None:
    """Complete per-part traceability after the common spawn translation is known."""

    translation = np.asarray(
        [
            -float(spawn["x_before_translation_m"]),
            -float(spawn["y_before_translation_m"]),
            -float(spawn["terrain_height_m"]),
        ],
        dtype=np.float64,
    )
    audit = simplification.get("part_geometry_audit")
    parts = simplification.get("parts")
    if not isinstance(audit, dict) or not isinstance(parts, list):
        raise FieldBuildError("逐零件几何审计结构缺失，无法记录最终世界坐标")
    for row in parts:
        if not isinstance(row, dict):
            raise FieldBuildError("逐零件几何审计记录类型无效")
        for phase in ("input", "output"):
            bounds = np.asarray(row[f"{phase}_bounds_m"], dtype=np.float64)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                raise FieldBuildError("逐零件几何审计边界无效")
            row[f"{phase}_bounds_after_spawn_translation_m"] = (bounds + translation).tolist()
    audit["final_world_translation_m"] = translation.tolist()
    audit["final_world_bounds_recorded"] = True


def _verified_repack_input(base_build: Path) -> tuple[dict[str, Any], Path, str, bytes]:
    base = Path(base_build).expanduser().resolve()
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise FieldBuildError(f"基础场地产物缺少manifest：{manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    source = manifest.get("source")
    if (
        manifest.get("artifact_type") != "rmuc2026_official_field_mujoco_asset"
        or manifest.get("status") != "PASS"
        or not isinstance(source, dict)
        or source.get("sha256") != OFFICIAL_STEP_SHA256
        or source.get("size_bytes") != OFFICIAL_STEP_SIZE
        or source.get("step_product") != EXPECTED_STEP_PRODUCT
    ):
        raise FieldBuildError("重打包输入未绑定通过的官方RMUC 2026 V2.0.0 Finals STEP")
    conversion = manifest.get("conversion")
    if not isinstance(conversion, dict):
        raise FieldBuildError("重打包输入缺少转换合同")
    intermediate = conversion.get("colored_intermediate_glb")
    if not isinstance(intermediate, dict) or not isinstance(intermediate.get("file"), str):
        raise FieldBuildError("重打包输入缺少官方中间GLB记录")
    glb_path = (base / str(intermediate["file"])).resolve()
    if base not in glb_path.parents or not glb_path.is_file():
        raise FieldBuildError(f"官方中间GLB缺失或越界：{glb_path}")
    if glb_path.stat().st_size != int(intermediate.get("size_bytes", -1)):
        raise FieldBuildError("官方中间GLB大小与基础manifest不一致")
    glb_sha256 = sha256_file(glb_path)
    if glb_sha256 != intermediate.get("sha256"):
        raise FieldBuildError("官方中间GLB SHA-256与基础manifest不一致")
    return manifest, glb_path, manifest_sha256, manifest_bytes


def _apply_verified_recorded_transform(
    meshes: list[Any], manifest: dict[str, Any]
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    conversion = manifest.get("conversion")
    if not isinstance(conversion, dict):
        raise FieldBuildError("基础场地产物缺少转换合同")
    axis = conversion.get("axis_and_units")
    if not isinstance(axis, dict):
        raise FieldBuildError("基础场地产物缺少轴/单位合同")
    order = np.asarray(axis.get("axis_order_output_xyz"), dtype=np.int64)
    scale = float(axis.get("unit_scale_to_metres", math.nan))
    recorded_input_bounds = np.asarray(axis.get("input_bounds"), dtype=np.float64)
    recorded_output_extents = np.asarray(
        axis.get("output_extents_before_ground_shift_m"), dtype=np.float64
    )
    if (
        order.shape != (3,)
        or sorted(order.tolist()) != [0, 1, 2]
        or not math.isfinite(scale)
        or scale <= 0.0
        or recorded_input_bounds.shape != (2, 3)
        or recorded_output_extents.shape != (3,)
        or not np.isfinite(recorded_input_bounds).all()
        or not np.isfinite(recorded_output_extents).all()
    ):
        raise FieldBuildError("基础场地产物轴/单位合同损坏")
    loaded_metrics = _mesh_set_metrics(meshes)
    loaded_bounds = np.asarray(loaded_metrics["bounds_m"], dtype=np.float64)
    tolerance = max(1.0e-6, 1.0e-6 * float(np.max(np.abs(recorded_input_bounds))))
    if not np.allclose(loaded_bounds, recorded_input_bounds, rtol=0.0, atol=tolerance):
        raise FieldBuildError(
            "官方中间GLB边界与记录不一致："
            f"max_abs={float(np.max(np.abs(loaded_bounds - recorded_input_bounds))):.6g}"
        )
    _apply_axis_contract(meshes, axis)
    transformed_metrics = _mesh_set_metrics(meshes)
    if not np.allclose(
        np.asarray(transformed_metrics["extents_m"], dtype=np.float64),
        recorded_output_extents,
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise FieldBuildError("应用记录轴/单位后，官方中间GLB尺寸不一致")
    recorded_ground = float(conversion.get("main_floor_height_before_shift_m", math.nan))
    if not math.isfinite(recorded_ground):
        raise FieldBuildError("基础场地产物主地面高度无效")
    measured_ground = _ground_height(meshes)
    if abs(measured_ground - recorded_ground) > 0.02:
        raise FieldBuildError(
            "官方中间GLB主地面高度与记录不一致："
            f"recorded={recorded_ground:.6f}, measured={measured_ground:.6f}"
        )
    for mesh in meshes:
        mesh.apply_translation([0.0, 0.0, -recorded_ground])
    transform_audit = {
        "recorded_contract_applied": True,
        "loaded_raw_geometry": loaded_metrics,
        "measured_main_floor_height_before_shift_m": measured_ground,
        "recorded_main_floor_height_before_shift_m": recorded_ground,
        "ground_height_abs_error_m": abs(measured_ground - recorded_ground),
    }
    return axis, recorded_ground, transform_audit


def _complete_collision_and_spawn(
    collision: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    height: np.ndarray,
) -> dict[str, float]:
    spawn = _choose_spawn(x, y, height)
    collision["geom_center_after_translation_m"] = [
        0.5 * (float(x[0]) + float(x[-1])) - spawn["x_before_translation_m"],
        0.5 * (float(y[0]) + float(y[-1])) - spawn["y_before_translation_m"],
        -spawn["terrain_height_m"],
    ]
    collision["half_size_xy_m"] = [
        0.5 * (float(x[-1]) - float(x[0])),
        0.5 * (float(y[-1]) - float(y[0])),
    ]
    collision["base_depth_m"] = 0.05
    spawn["x_after_translation_m"] = 0.0
    spawn["y_after_translation_m"] = 0.0
    spawn["terrain_height_after_translation_m"] = 0.0
    return spawn


def _translated_bounds(metrics: dict[str, Any], spawn: dict[str, float]) -> list[list[float]]:
    bounds = np.asarray(metrics["bounds_m"], dtype=np.float64)
    translation = np.asarray(
        [
            -float(spawn["x_before_translation_m"]),
            -float(spawn["y_before_translation_m"]),
            -float(spawn["terrain_height_m"]),
        ],
        dtype=np.float64,
    )
    return (bounds + translation).tolist()


def repack_field_build(
    base_build: Path,
    output_dir: Path,
    *,
    target_visual_faces: int = 450_000,
    heightfield_resolution_m: float = RECOMMENDED_HEIGHTFIELD_RESOLUTION_M,
) -> dict[str, Any]:
    """Fast, provenance-checked rebuild from an existing official OpenCascade GLB."""

    import trimesh

    if target_visual_faces < 50_000:
        raise FieldBuildError("视觉网格面数预算过低，无法辨认官方场地")
    if not MINIMUM_HEIGHTFIELD_RESOLUTION_M <= heightfield_resolution_m <= 0.25:
        raise FieldBuildError(
            f"高度图分辨率必须在[{MINIMUM_HEIGHTFIELD_RESOLUTION_M:.02f}, 0.25]米"
        )
    base = Path(base_build).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FieldBuildError(f"输出目录已存在，拒绝覆盖：{output}")
    manifest, glb_path, base_manifest_sha256, base_manifest_bytes = _verified_repack_input(base)
    started = time.monotonic()
    output.mkdir(parents=True)
    copied_glb = output / "official_rmuc2026_colored.glb"
    shutil.copy2(glb_path, copied_glb)
    copied_glb_sha256 = sha256_file(copied_glb)
    if copied_glb_sha256 != sha256_file(glb_path):
        raise FieldBuildError("官方中间GLB独立复制后哈希变化")

    scene = trimesh.load(glb_path, force="scene", process=False)
    raw_meshes = _scene_meshes(scene)
    axis, ground, transform_audit = _apply_verified_recorded_transform(raw_meshes, manifest)
    raw_transformed_metrics = _mesh_set_metrics(raw_meshes)
    visual_records, final_meshes, simplification = _simplify_parts_then_group(
        raw_meshes,
        output,
        target_faces=target_visual_faces,
    )
    collision, x, y, height = _ray_heightfield(
        raw_meshes,
        output,
        resolution_m=heightfield_resolution_m,
        source_glb_sha256=copied_glb_sha256,
    )
    collision["geometry_source"] = (
        "raw_axis_and_ground_transformed_official_glb_before_visual_simplification"
    )
    collision["source_faces"] = int(raw_transformed_metrics["faces"])
    collision["source_surface_area_m2"] = float(raw_transformed_metrics["surface_area_m2"])
    spawn = _complete_collision_and_spawn(collision, x, y, height)
    collision["fixed_fly_ramp_audit"] = audit_fixed_fly_ramps(x, y, height)
    _translate_final_assets(final_meshes, visual_records, output, spawn=spawn)
    _record_part_geometry_world_bounds(simplification, spawn=spawn)
    translated_outer_bounds = _translated_bounds(raw_transformed_metrics, spawn)

    base_conversion = manifest.get("conversion")
    assert isinstance(base_conversion, dict)
    result: dict[str, Any] = {
        "schema_version": 3,
        "artifact_type": "rmuc2026_official_field_mujoco_asset",
        "build_variant": "official_glb_per_original_part_repack_v1",
        "status": "PASS",
        "source": manifest["source"],
        "repacked_from": {
            "path_at_build": str(base),
            "manifest_sha256": base_manifest_sha256,
            "input_glb": {
                "path_at_build": str(glb_path),
                "sha256": copied_glb_sha256,
                "size_bytes": copied_glb.stat().st_size,
            },
            "base_manifest_was_not_modified": True,
        },
        "cad_audit": manifest.get("cad_audit"),
        "conversion": {
            "backend": "existing_official_OpenCascade_GLTF_fast_repack",
            "source_backend": base_conversion.get("backend"),
            "source_linear_deflection_mm": base_conversion.get("linear_deflection_mm"),
            "source_angular_deflection_rad": base_conversion.get("angular_deflection_rad"),
            "target_visual_faces": target_visual_faces,
            "cad_appearance_imported": base_conversion.get("cad_appearance_imported"),
            "cad_color_output": base_conversion.get("cad_color_output"),
            "axis_and_units": axis,
            "main_floor_height_before_shift_m": ground,
            "recorded_transform_audit": transform_audit,
            "colored_intermediate_glb": {
                "file": copied_glb.name,
                "sha256": copied_glb_sha256,
                "size_bytes": copied_glb.stat().st_size,
                "copy_mode": "independent_byte_copy",
            },
            "visual_simplification": simplification,
        },
        "visual_meshes": sorted(visual_records, key=lambda row: str(row["material_id"])),
        "collision": collision,
        "validation_scope": _static_route_validation_scope(collision),
        "recommended_spawn": spawn,
        "dimensions": {
            "official_core_battlefield_m": list(OFFICIAL_CORE_BATTLEFIELD_SIZE_M),
            "official_rulebook_url": OFFICIAL_RULEBOOK_URL,
            "cad_assembly_outer_extents_m": raw_transformed_metrics["extents_m"],
            "cad_assembly_outer_bounds_after_translation_m": translated_outer_bounds,
            "interpretation": (
                "28x15 m is the playable core; visual diagnostic illustrations map to the "
                "larger official CAD assembly outer rectangle including fences/protrusions"
            ),
        },
        "wall_seconds": time.monotonic() - started,
        "evidence_boundary": {
            "official_visual_geometry": True,
            "visual_geometry_simplified_per_original_part": True,
            "exact_brep_collision": False,
            "reference_wheel_spatial_sampling_ready": bool(
                collision["resolution_quality"]["fine_interaction_spatial_sampling_ready"]
            ),
            "multilevel_collision_topology_supported": False,
            "realtime_collision": (
                "raw official GLB-derived single-valued 2.5D top-surface heightfield proxy"
            ),
            "movable_props_dynamic": False,
            "rulebook_surface_guide_present": False,
        },
    }
    _write_json_atomically(output / "manifest.json", result)
    if (base / "manifest.json").read_bytes() != base_manifest_bytes:
        raise FieldBuildError("重打包意外修改了基础manifest")
    if sha256_file(glb_path) != copied_glb_sha256:
        raise FieldBuildError("重打包期间基础官方GLB发生变化")
    print("RMUC2026_FIELD_REPACK=PASS", flush=True)
    return result


def build_field(
    step_path: Path,
    output_dir: Path,
    *,
    linear_deflection_mm: float = 30.0,
    angular_deflection_rad: float = 0.5,
    target_visual_faces: int = 450_000,
    heightfield_resolution_m: float = RECOMMENDED_HEIGHTFIELD_RESOLUTION_M,
    preserve_cad_colors: bool = True,
) -> dict[str, Any]:
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.Message import Message_ProgressRange
    from OCP.RWGltf import RWGltf_CafWriter
    from OCP.TCollection import TCollection_AsciiString
    from OCP.TColStd import TColStd_IndexedDataMapOfStringString
    import trimesh

    if linear_deflection_mm <= 0.0 or angular_deflection_rad <= 0.0:
        raise FieldBuildError("网格化公差必须为正数")
    if target_visual_faces < 50_000:
        raise FieldBuildError("视觉网格面数预算过低，无法辨认官方场地")
    if not MINIMUM_HEIGHTFIELD_RESOLUTION_M <= heightfield_resolution_m <= 0.25:
        raise FieldBuildError(
            f"高度图分辨率必须在[{MINIMUM_HEIGHTFIELD_RESOLUTION_M:.02f}, 0.25]米"
        )
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FieldBuildError(f"输出目录已存在，拒绝覆盖：{output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    identity = validate_official_step(step_path, verify_hash=True)
    if preserve_cad_colors:
        document, shape, audit = _load_xcaf(identity.path)
        audit["appearance_metadata_preserved"] = True
    else:
        document, shape, audit = _load_geometry_only(identity.path)
    print("RMUC2026_FIELD_STEP_TRANSFER=PASS", flush=True)

    mesher = BRepMesh_IncrementalMesh(
        shape,
        float(linear_deflection_mm),
        False,
        float(angular_deflection_rad),
        True,
    )
    if not mesher.IsDone():
        raise FieldBuildError("OpenCascade场地网格化未完成")
    print("RMUC2026_FIELD_MESHING=PASS", flush=True)
    intermediate = output / "official_rmuc2026_colored.glb"
    writer = RWGltf_CafWriter(TCollection_AsciiString(str(intermediate)), True)
    writer.SetParallel(True)
    writer.SetMergeFaces(True)
    metadata = TColStd_IndexedDataMapOfStringString()
    if not writer.Perform(document, metadata, Message_ProgressRange()):
        raise FieldBuildError("OpenCascade无法导出带装配颜色的glTF")
    if not intermediate.is_file() or intermediate.stat().st_size == 0:
        raise FieldBuildError("OpenCascade未生成有效glTF")
    print("RMUC2026_FIELD_GLTF=PASS", flush=True)
    intermediate_hash = sha256_file(intermediate)

    scene = trimesh.load(intermediate, force="scene", process=False)
    meshes = _scene_meshes(scene)
    axis = _axis_contract(meshes)
    _apply_axis_contract(meshes, axis)
    ground = _ground_height(meshes)
    for mesh in meshes:
        mesh.apply_translation([0.0, 0.0, -ground])
    raw_transformed_metrics = _mesh_set_metrics(meshes)
    visual_records, final_meshes, simplification = _simplify_parts_then_group(
        meshes,
        output,
        target_faces=target_visual_faces,
    )
    collision, x, y, height = _ray_heightfield(
        meshes,
        output,
        resolution_m=heightfield_resolution_m,
        source_glb_sha256=intermediate_hash,
    )
    collision["geometry_source"] = (
        "raw_axis_and_ground_transformed_official_glb_before_visual_simplification"
    )
    collision["source_faces"] = int(raw_transformed_metrics["faces"])
    collision["source_surface_area_m2"] = float(raw_transformed_metrics["surface_area_m2"])
    spawn = _complete_collision_and_spawn(collision, x, y, height)
    collision["fixed_fly_ramp_audit"] = audit_fixed_fly_ramps(x, y, height)
    _translate_final_assets(final_meshes, visual_records, output, spawn=spawn)
    _record_part_geometry_world_bounds(simplification, spawn=spawn)
    translated_outer_bounds = _translated_bounds(raw_transformed_metrics, spawn)
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "artifact_type": "rmuc2026_official_field_mujoco_asset",
        "build_variant": "official_step_per_original_part_v1",
        "status": "PASS",
        "source": identity.as_dict(),
        "cad_audit": audit,
        "conversion": {
            "backend": "OpenCascade/OCP",
            "linear_deflection_mm": linear_deflection_mm,
            "angular_deflection_rad": angular_deflection_rad,
            "target_visual_faces": target_visual_faces,
            "cad_appearance_imported": preserve_cad_colors,
            "cad_color_output": (
                "source_material_colors_preserved_per_rgba_group"
                if preserve_cad_colors
                else "single_gray_fallback"
            ),
            "axis_and_units": axis,
            "main_floor_height_before_shift_m": ground,
            "colored_intermediate_glb": {
                "file": intermediate.name,
                "sha256": intermediate_hash,
                "size_bytes": intermediate.stat().st_size,
            },
            "visual_simplification": simplification,
        },
        "visual_meshes": sorted(visual_records, key=lambda row: str(row["material_id"])),
        "collision": collision,
        "validation_scope": _static_route_validation_scope(collision),
        "recommended_spawn": spawn,
        "dimensions": {
            "official_core_battlefield_m": list(OFFICIAL_CORE_BATTLEFIELD_SIZE_M),
            "official_rulebook_url": OFFICIAL_RULEBOOK_URL,
            "cad_assembly_outer_extents_m": raw_transformed_metrics["extents_m"],
            "cad_assembly_outer_bounds_after_translation_m": translated_outer_bounds,
            "interpretation": (
                "28x15 m is the playable battlefield; the STEP assembly outer bounds "
                "also include fences and protruding construction"
            ),
        },
        "wall_seconds": time.monotonic() - started,
        "evidence_boundary": {
            "official_visual_geometry": True,
            "exact_brep_collision": False,
            "reference_wheel_spatial_sampling_ready": bool(
                collision["resolution_quality"]["fine_interaction_spatial_sampling_ready"]
            ),
            "multilevel_collision_topology_supported": False,
            "realtime_collision": "STEP-derived single-valued 2.5D heightfield proxy",
            "movable_props_dynamic": False,
        },
    }
    _write_json_atomically(output / "manifest.json", manifest)
    print("RMUC2026_FIELD_BUILD=PASS", flush=True)
    return manifest
