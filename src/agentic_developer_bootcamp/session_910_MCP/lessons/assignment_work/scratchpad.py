# Testing working of the sample code

for question in (
        "Order 8812 arrived damaged. I paid Rs 3000 total. Process my refund.",
        "Courier is stuck. What is the weather in Bangalore right now?",
        "What is a chargeback, in one sentence?",
    ):
        print({"thread_id": question[:40]})