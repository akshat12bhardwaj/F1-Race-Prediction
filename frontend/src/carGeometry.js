// Source-specific corrections are pinned to the unchanged Formula F222 GLB.
// The open cockpit and intake boundaries are intentional; never fill or double-side them.
export const F222_REPAIRS = {
  reverse: new Set(['Object_60', 'Object_64', 'Object_102', 'Object_112', 'Object_29', 'Object_42', 'Object_44', 'Object_46', 'Object_50', 'Object_52', 'Object_86', 'Object_88', 'Object_90', 'Object_94', 'Object_96']),
  windingOnly: new Set(['Object_104', 'Object_110']),
  normalsOnly: new Set(['Object_81']),
}

export function orientCarGeometry(geometry, name, matrix, repairF222 = false) {
  const reverse = repairF222 && F222_REPAIRS.reverse.has(name)
  const windingOnly = repairF222 && F222_REPAIRS.windingOnly.has(name)
  const normalsOnly = repairF222 && F222_REPAIRS.normalsOnly.has(name)
  // Baking removes Three's draw-time determinant test; preserve reflected mesh front faces.
  if ((matrix.determinant() < 0) !== !!(reverse || windingOnly)) {
    const index = geometry.index
    if (index) for (let i = 0; i < index.count; i += 3) { const b = index.getX(i + 1); index.setX(i + 1, index.getX(i + 2)); index.setX(i + 2, b) }
    else for (const attribute of Object.values(geometry.attributes)) for (let i = 0; i < attribute.count; i += 3) for (let k = 0; k < attribute.itemSize; k++) {
      const b = attribute.array[(i + 1) * attribute.itemSize + k]
      attribute.array[(i + 1) * attribute.itemSize + k] = attribute.array[(i + 2) * attribute.itemSize + k]
      attribute.array[(i + 2) * attribute.itemSize + k] = b
    }
  }
  if (reverse || normalsOnly) {
    const normal = geometry.attributes.normal
    for (let i = 0; i < normal.count; i++) normal.setXYZ(i, -normal.getX(i), -normal.getY(i), -normal.getZ(i))
  }
  return geometry.applyMatrix4(matrix)
}
