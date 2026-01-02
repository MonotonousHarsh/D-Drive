# pickleball_scheduler/tools.py

import datetime

def list_courts_availability(date: str, start_time: str = None) -> str:
    """
    Checks the availability of pickleball courts for a given date and optional start time.
    In a real application, this would call an external API.
    
    Args:
        date (str): The date to check for availability in YYYY-MM-DD format.
        start_time (str, optional): The specific start time to check in HH:MM format. Defaults to None.
        
    Returns:
        str: A formatted string listing available court times and IDs.
    """
    print(f"[*] Tool: Checking court availability for {date} around {start_time or 'any time'}...")
    # Mock data for demonstration
    mock_courts = {
        "court_1": ["15:00", "16:00", "17:00", "18:00"],
        "court_2": ["16:30", "17:30", "18:30"],
        "court_3": ["15:00", "18:00"],
    }
    
    # Simple filtering logic
    available_slots = []
    for court, times in mock_courts.items():
        for t in times:
            if start_time is None or t.startswith(start_time.split(':')[0]):
                available_slots.append(f"  - Court ID: {court} at {t}")

    if not available_slots:
        return f"No courts are available on {date} around {start_time}."
        
    return f"Available courts on {date}:\n" + "\n".join(available_slots)


def book_court(court_id: str, date: str, time: str, player_names: list[str]) -> str:
    """
    Books a specific pickleball court for a given date, time, and list of players.
    In a real application, this would call a booking API.

    Args:
        court_id (str): The ID of the court to book.
        date (str): The date of the booking in YYYY-MM-DD format.
        time (str): The time of the booking in HH:MM format.
        player_names (list[str]): A list of the names of the players.

    Returns:
        str: A confirmation message with the booking details.
    """
    print(f"[*] Tool: Attempting to book {court_id} on {date} at {time} for {', '.join(player_names)}...")
    # Mock confirmation
    confirmation_id = f"BKNG-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    return (
        f"Success! Booking confirmed.\n"
        f"Confirmation ID: {confirmation_id}\n"
        f"Court: {court_id}\n"
        f"Date: {date}\n"
        f"Time: {time}\n"
        f"Players: {', '.join(player_names)}"
    )