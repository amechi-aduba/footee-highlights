/** Human clip name: "seg_003" -> "Clip 3". Backend IDs stay stable. */
export function clipLabel(segmentId: string) {
  const match = /^seg_(\d+)$/.exec(segmentId);
  return match ? `Clip ${parseInt(match[1], 10)}` : segmentId;
}
