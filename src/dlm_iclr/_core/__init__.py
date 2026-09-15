"""Crystal-generation research package.

The active source-only distribution contains the stratified Wyckoff
co-diffusion package.  Historical fixed-slot modules remain in the local
archive/source tree for provenance, but the package root must not eagerly
import them: doing so would make a clean source-bundle installation depend on
files that are deliberately excluded from the active execution manifest.
"""

__all__: list[str] = []
