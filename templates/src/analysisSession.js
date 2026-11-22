export async function loadPredictionModel(request, { signal, wait, maxAttempts = 5 }) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    let response;
    let data;
    try {
      response = await request(signal);
      data = await response.json();
    } catch (error) {
      if (signal.aborted) throw error;
      data = { retryable: true, message: "The prediction service could not be reached." };
    }
    if (signal.aborted) return null;
    if (response?.ok && data.status === "ready") return data;
    const retryable = data.retryable || response?.status === 429 || response?.status >= 500;
    if (!retryable || attempt + 1 === maxAttempts) {
      return { status: data.status || "unavailable", message: data.message || data.error || "Prediction model could not be loaded." };
    }
    await wait(300 * 2 ** attempt, signal);
    if (signal.aborted) return null;
  }
  return null;
}

export function waitForRetry(ms, signal) {
  return new Promise(resolve => {
    if (signal.aborted) { resolve(); return; }
    const finish = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", finish);
      resolve();
    };
    const timer = setTimeout(finish, ms);
    signal.addEventListener("abort", finish, { once: true });
  });
}
