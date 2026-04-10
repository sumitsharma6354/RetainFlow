def get_retention_action(customer: dict, churn_prob: float, segment: str) -> dict:
    contract = customer.get('Contract', '')
    monthly_charges = float(customer.get('MonthlyCharges', 0))
    tenure = int(customer.get('tenure', 0))
    internet_service = customer.get('InternetService', '')

    offer = ""
    urgency = ""
    
    if churn_prob >= 0.7 and contract == "Month-to-month":
        offer = "20% discount on annual plan upgrade"
        urgency = "critical"
    elif churn_prob >= 0.7 and monthly_charges > 70:
        offer = "Free premium bundle for 3 months"
        urgency = "critical"
    elif churn_prob >= 0.5 and tenure < 12:
        offer = "Dedicated onboarding specialist + free tech support for 6 months"
        urgency = "high"
    elif churn_prob >= 0.5 and internet_service == "Fiber optic":
        offer = "Upgrade to 1Gbps at current price for 12 months"
        urgency = "high"
    elif churn_prob >= 0.4:
        offer = "Loyalty reward: $10 credit added to account"
        urgency = "medium"
    else:
        offer = "Send satisfaction survey + highlight unused features"
        urgency = "low"

    return {
        "offer": offer,
        "urgency": urgency,
        "estimated_save_value": float(monthly_charges * 12)
    }
