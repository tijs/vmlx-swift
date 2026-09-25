# Native MTP default Off checkpoint — PARTIAL

## Superseded scope — September 14

The September 13 global-Off proposal below is superseded by Eric's request:
**Flash Next only starts Off; 27B must not be affected.** Shared engine init
and missing-mode decoding therefore remain Auto. Osaurus applies the family
default from top-level/nested `model_type=qwen4_exp`, independently of quant
name and selector capability. Real heads retain Off/Auto/D1/D2/D3 controls;
explicit choices are preserved. Family-owned Off is reversed on leaving Flash
Next, including restoration of the existing eligible Qwen27B D3 picker default.
This is not a ban on manually activating Flash Next MTP, nor a change to API
callers explicitly requesting Auto. No sampler or activation gate is weakened.

The new scoped app tests/build/UI receipts are pending. September 13 global-Off
tests below are historical evidence, not proof of the corrected policy.

## Historical proposal (not the current product policy)

The requested product policy is AR until a user selects Auto or a depth.
MTP capability remains a property of bundle tensors, independent of activation.

## Source and scope

`VMLXServerMTPSettings.init` and missing-mode decoding now default to `.off`.
`resolvedMTPLaunch`, `resolvedLoadConfiguration`, and `resolvedMTPDraftStrategy`
retain their existing explicit activation and tensor/tuning validation.
An explicitly configured DFlash drafter remains a separate user choice.
No sampler, cache topology, precision, kernel, or performance setting changes.

The companion Osaurus change removes picker-triggered D3 activation and the
obsolete legacy Off-to-Auto migration. Its new one-shot migration recognizes
only untouched old Auto or recorded family-owned D3. Recorded explicit choices,
custom draft/cache fields and later document edits are preserved. A previous
explicit Auto with no provenance and exactly factory-shaped fields cannot be
distinguished from untouched Auto; that shape is migrated once.

## Executed engine evidence

Local M5 Max2, September 13. `MTPDefaultOffEngineFinal0913__181630`:
40 Swift Testing cases in two suites, exit 0, cleanup group/tracked/watchdog
0/0/0, peak supervised footprint 1.83 GiB, swap 0.49 GiB unchanged.
Suite filter: `VMLXServerRuntimeSettingsTests`. Covers default/missing-mode Off,
complete measured heads remaining AR, explicit Auto resolution, native-load
override, sampler top-k wiring, and existing settings validation.
Tests use optimized Swift with a pre-existing DEBUG-only test hook enabled;
that hook is not part of the development-app build configuration.

Earlier attempts are retained: `180206` failed to compile the existing
Release-only lock-test mismatch; `180528` exposed an old legacy-4M fixture
missing the topology fingerprint required by the current gate. Only that
fixture was completed; no runtime block or safety validation was relaxed.

Raw logs are under the private `mtp-swift-2026-09-04/logs` evidence directory.
App pinning, focused app tests, rebuilt binary and live picker/relaunch/AR
execution are still required. This document is not app or speed proof.
