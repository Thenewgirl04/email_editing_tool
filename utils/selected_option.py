def option_to_action(option):
    task_label_to_key = {
        "Shorten an email": "shorten",
        "Lengthen an email": "lengthen",
        "Professional": "professional",
        "Friendly": "friendly",
        "Sympathetic": "sympathetic",
    }
    return task_label_to_key.get(option)
