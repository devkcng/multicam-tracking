import axios, { AxiosInstance, AxiosRequestConfig } from "axios";

const axiosConfig: AxiosRequestConfig = {
  baseURL: "https://above-ruling-ringtail.ngrok-free.app",
  headers: {
    Accept: "application/json",
    "ngrok-skip-browser-warning": "true",
  },
};

// Tạo instance của Axios với cấu hình mặc định
const apiClient: AxiosInstance = axios.create(axiosConfig);

export default apiClient;
