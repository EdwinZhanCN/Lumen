import { ApiError } from "@/lib/api";

export type ErrorPresentation = {
  title: string;
  message: string;
};

export function describeUiError(
  error: unknown,
  fallbackMessage: string,
): ErrorPresentation {
  if (error instanceof ApiError) {
    if (error.kind === "network") {
      return {
        title: "网络错误",
        message: error.message || fallbackMessage,
      };
    }
    if (error.kind === "permission") {
      return {
        title: "权限错误",
        message: error.message || fallbackMessage,
      };
    }
    if (error.kind === "business") {
      return {
        title: "请求参数错误",
        message: error.message || fallbackMessage,
      };
    }
    if (error.kind === "server") {
      return {
        title: "服务内部错误",
        message: error.message || fallbackMessage,
      };
    }
  }

  if (error instanceof Error) {
    return {
      title: "请求失败",
      message: error.message || fallbackMessage,
    };
  }

  return {
    title: "未知错误",
    message: fallbackMessage,
  };
}
