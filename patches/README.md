# Optional patches for the i3dgs submodule

The submodule is pinned and unmodified; nothing here is applied. Each patch is a plain
`git diff` against cf4d5b9 and applies with git from inside the submodule.

    git -C submodules/i3dgs apply ../../patches/<name>.diff       # apply
    git -C submodules/i3dgs apply -R ../../patches/<name>.diff    # revert
    git -C submodules/i3dgs apply --check ../../patches/<name>.diff   # dry run
    git -C submodules/i3dgs checkout -- .                         # discard everything

## i3dgs_defer_syncs.diff

Removes about 6200 device-to-host reads per scene: the per-candidate `.item()` loops in
loop closure become one `.tolist()`, and the bundle-adjustment observation counter is read
once per match loop instead of once per match. Measured at 0.4% on four scenes, so it is
kept here rather than carried.

The counter deferral cannot span the whole of `append`: the host block writes new-landmark
observations at `s = self.obs_size`, so an over-estimate there lands them at the wrong
offset and silently corrupts observations. The patch therefore reads the true count at both
loop boundaries and re-seeds the device counter between them.
