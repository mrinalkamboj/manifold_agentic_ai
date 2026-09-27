"""Concept: tools live in a module. Another file will import TOOLS."""
from langchain_core.tools import tool


@tool
def calculate_incident_impact(failed_requests: int, total_requests: int, baseline_error_rate_pct: float) -> dict:
    """Calculate request failure percentage and percentage-point change from baseline.
    Use counts from the same observation window.
    """
    rate = failed_requests / total_requests * 100
    return {"error_rate_pct": rate, "change_percentage_points": rate - baseline_error_rate_pct}


@tool
def lookup_runbook(service: str) -> str:
    """Return the local runbook steps for a known service. service is a short name like checkout or login."""
    runbooks = {
        "checkout": "1. Check payment gateway. 2. Compare error rate to baseline. 3. Page payments if change is over 2 points.",
        "login": "1. Check auth service. 2. Confirm password reset is up. 3. Log a ticket if only one user is affected.",
    }
    return runbooks.get(service.lower(), f"No local runbook for {service}.")


TOOLS = [calculate_incident_impact, lookup_runbook]


if __name__ == "__main__":
    print(calculate_incident_impact.invoke({
        "failed_requests": 60,
        "total_requests": 1200,
        "baseline_error_rate_pct": 2.0,
    }))
    print(lookup_runbook.invoke({"service": "checkout"}))
    print("Who selected the function and arguments? The developer did. The graph is not involved yet.")
