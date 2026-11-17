# Glossary

**Car Photo** (รูปรถยนต์)
The image the user uploads. May contain zero, one, or many cars.

**Plate** (ป้ายทะเบียน)
A Thai vehicle license plate visible in a Car Photo. One Car Photo can hold many Plates; every Plate is read.
_Avoid_: license, tag

**Plate Crop** (ภาพป้ายที่ตัด)
The region of the Car Photo cut out around one detected Plate. Shown to the user next to its Reading.

**Registration** (ทะเบียน)
The main line of a Plate: digits and Thai letters, e.g. `1กข 1234`. The primary thing this project must read correctly.

**Province** (จังหวัด)
The smaller line under the Registration naming a Thai province, e.g. `กรุงเทพมหานคร`. Read best-effort; may be missing or wrong. Always one of Thailand's 77 provinces, or empty.

**Reading** (ผลการอ่าน)
The text read from one Plate Crop: a Registration plus an optional Province.

**Detection Confidence** (ความมั่นใจการตรวจจับ)
How sure the detector is that a region is a Plate (0–1).

**Reading Confidence** (ความมั่นใจการอ่าน)
How sure the reader is about a Reading's text (0–1). Independent of Detection Confidence: a clearly found Plate can still be read poorly.

**Exact Match** (อ่านถูกทั้งหมด)
A Reading whose Registration equals the hand-typed true Registration character for character. The headline accuracy measure.
