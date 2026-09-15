# Third-party notices

The MIT license in `LICENSE` applies only to the original Python code in this
package. It does not grant rights to RoboMaster, RMUC, DJI, MuJoCo, the RMUC
2026 STEP model, the competition rulebook, or files derived from those works.

The loader recognizes a local field asset whose provenance records the
following official-source identity:

- product: `00_RMUC2026_FINALS_ASM`;
- filename: `RMUC2026_V2.0.0.stp`;
- size: `1,254,821,405` bytes;
- SHA-256: `8dfe9ebd761e44d91361b3e593bc05416329112217b58cb35800b3cde2ffae33`;
- official publication page: <https://bbs.robomaster.com/article/814728?source=8>.

No official file or derived binary asset is included in this package. The
upstream publication being publicly downloadable is not treated here as a
redistribution license. Users and distributors are responsible for obtaining
the relevant permissions and following upstream terms.

Fudan wheel-legged robot material is a separate upstream work. This package
does not include its URDF/MJCF files, meshes, training code, checkpoints,
exported policies, or namespaced derivatives. A user's robot model is supplied
independently when composing a scene and retains its own license.

RoboMaster, RMUC, and DJI names and marks belong to their respective owners.
This project is community-authored, unofficial, and is not endorsed by or
affiliated with DJI, RoboMaster, the RMUC organizers, or Fudan University.

MuJoCo is an independent third-party dependency distributed under its own
license. NumPy and the selected Python build backend likewise retain their own
licenses. See the installed distributions for their notices.
