const SERVICE_NAME_PATTERN =
  /^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$/;

export function isValidPort(port: number): boolean {
  return Number.isInteger(port) && port >= 1024 && port <= 65535;
}

export function getPortValidationMessage(rawPort: string): string | null {
  const trimmed = rawPort.trim();
  if (!trimmed) {
    return "端口号不能为空";
  }
  if (!/^\d+$/.test(trimmed)) {
    return "端口号必须为数字";
  }
  const parsed = Number.parseInt(trimmed, 10);
  if (!isValidPort(parsed)) {
    return "端口号必须在 1024 - 65535 之间";
  }
  return null;
}

export function parseValidPort(rawPort: string): number | null {
  return getPortValidationMessage(rawPort) ? null : Number.parseInt(rawPort, 10);
}

export function isValidServiceName(serviceName: string): boolean {
  const trimmed = serviceName.trim();
  if (trimmed.length < 3 || trimmed.length > 63) {
    return false;
  }
  return SERVICE_NAME_PATTERN.test(trimmed);
}

export function getServiceNameValidationMessage(
  serviceName: string,
): string | null {
  const trimmed = serviceName.trim();
  if (!trimmed) {
    return "服务名称不能为空";
  }
  if (trimmed.length < 3 || trimmed.length > 63) {
    return "服务名称长度必须在 3 - 63 之间";
  }
  if (!SERVICE_NAME_PATTERN.test(trimmed)) {
    return "服务名称仅支持字母、数字和中划线，且不能以中划线开头或结尾";
  }
  return null;
}
