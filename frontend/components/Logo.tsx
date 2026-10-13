import Image from "next/image";

/** The site mark. Served as-is from /public: the file is already small, so no image optimizer is needed. */
export function Logo({ size = 28 }: { size?: number }) {
  return <Image src="/logo.webp" alt="" width={size} height={size} unoptimized priority className="shrink-0 rounded-full" />;
}
