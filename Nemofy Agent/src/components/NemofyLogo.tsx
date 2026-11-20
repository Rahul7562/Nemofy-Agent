interface NemofyLogoProps {
  size?: number;
  className?: string;
}

/**
 * Exact 4-block Nemofy brand mark synced with https://nemofy.tech/
 */
export function NemofyLogo({ size = 28, className = '' }: NemofyLogoProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 160 160"
      width={size}
      height={size}
      className={className}
      aria-hidden="true"
    >
      <g transform="translate(20, 22)">
        <rect x="0" y="0" width="120" height="30" rx="10" fill="#473bf0" />
        <rect x="0" y="43" width="86" height="30" rx="10" fill="#f64b4b" />
        <rect x="0" y="86" width="44" height="30" rx="10" fill="#f59e0b" />
        <circle cx="102" cy="101" r="16" fill="#10b981" />
      </g>
    </svg>
  );
}
