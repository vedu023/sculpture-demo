from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Any, Callable


CalendarEvent = dict[str, Any]
Runner = Callable[..., subprocess.CompletedProcess[str]]

_APPLE_SCRIPT = """
on padNumber(valueNumber)
    return text -2 thru -1 of ("0" & (valueNumber as integer))
end padNumber

on sanitizeText(valueText)
    set cleanedText to valueText as text
    set AppleScript's text item delimiters to tab
    set cleanedText to text items of cleanedText
    set AppleScript's text item delimiters to " "
    set cleanedText to cleanedText as text
    set AppleScript's text item delimiters to linefeed
    set cleanedText to text items of cleanedText
    set AppleScript's text item delimiters to " "
    set cleanedText to cleanedText as text
    set AppleScript's text item delimiters to ""
    return cleanedText
end sanitizeText

on formatDateStamp(valueDate)
    set yearValue to year of valueDate as integer
    set monthValue to my padNumber(month of valueDate as integer)
    set dayValue to my padNumber(day of valueDate as integer)
    set hourValue to my padNumber(hours of valueDate as integer)
    set minuteValue to my padNumber(minutes of valueDate as integer)
    return (yearValue as string) & "-" & monthValue & "-" & dayValue & " " & hourValue & ":" & minuteValue
end formatDateStamp

on run argv
    set scopeName to "today"
    if (count of argv) > 0 then set scopeName to item 1 of argv

    set windowStart to current date
    set hours of windowStart to 0
    set minutes of windowStart to 0
    set seconds of windowStart to 0

    if scopeName is "tomorrow" then
        set windowStart to windowStart + (1 * days)
    end if

    set windowEnd to windowStart + (1 * days)
    set outputLines to {}

    tell application "Calendar"
        repeat with cal in calendars
            set calName to my sanitizeText(name of cal)
            set matchingEvents to every event of cal whose start date >= windowStart and start date < windowEnd
            repeat with ev in matchingEvents
                set eventTitle to my sanitizeText(summary of ev)
                set eventStart to my formatDateStamp(start date of ev)
                set eventEnd to my formatDateStamp(end date of ev)
                set eventLocation to ""
                try
                    set eventLocation to my sanitizeText(location of ev)
                end try
                set end of outputLines to calName & tab & eventTitle & tab & eventStart & tab & eventEnd & tab & eventLocation
            end repeat
        end repeat
    end tell

    if (count of outputLines) is 0 then
        return ""
    end if

    set AppleScript's text item delimiters to linefeed
    set outputText to outputLines as text
    set AppleScript's text item delimiters to ""
    return outputText
end run
""".strip()


def parse_calendar_output(output: str) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        while len(parts) < 5:
            parts.append("")
        calendar_name, title, start_text, end_text, location = parts[:5]
        events.append(
            {
                "calendar_name": calendar_name,
                "title": title,
                "start_text": start_text,
                "end_text": end_text,
                "location": location,
            }
        )
    return sorted(events, key=lambda event: (event["start_text"], event["end_text"], event["title"]))


@dataclass
class MacOSCalendarProvider:
    runner: Runner | None = None

    def __call__(self, date_scope: str) -> list[CalendarEvent]:
        active_runner = self.runner or subprocess.run
        result = active_runner(
            ["osascript", "-", date_scope],
            input=_APPLE_SCRIPT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error_text = result.stderr.strip() or result.stdout.strip() or "unknown Calendar access error"
            raise RuntimeError(error_text)
        return parse_calendar_output(result.stdout)
