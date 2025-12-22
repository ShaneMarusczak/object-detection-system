# Notification System

The object detection system includes a powerful notification feature that allows you to receive email or SMS alerts when specific event patterns occur. This is perfect for long-term monitoring scenarios, such as checking on pets while traveling.

## Overview

The notification system:
- ✅ Runs independently of console output (works in silent mode)
- ✅ Still logs all events to JSON files
- ✅ Sends alerts via email (SMTP) and/or SMS (Twilio)
- ✅ Supports pattern matching on event properties
- ✅ Includes cooldown periods to prevent spam
- ✅ Requires no code changes - configure via YAML

## Use Case Example

You're going out of town for 2 weeks and want to monitor your cats eating and drinking:

1. Set console output to `silent` mode (minimal logging)
2. Configure zones for food and water bowls
3. Set up notification rules to alert you when cats visit these areas
4. Run the system for extended periods
5. Receive periodic notifications confirming your cats are active

## Quick Start

### 1. Configure Zones in `config.yaml`

First, define zones for the areas you want to monitor:

```yaml
zones:
  - x1_pct: 10
    y1_pct: 20
    x2_pct: 30
    y2_pct: 40
    description: "food bowl"
    allowed_classes: [15, 16]  # cats and dogs

  - x1_pct: 40
    y1_pct: 20
    x2_pct: 60
    y2_pct: 40
    description: "water bowl"
    allowed_classes: [15, 16]
```

### 2. Enable Silent Mode (Optional but Recommended)

For long-running monitoring with minimal console output:

```yaml
console_output:
  enabled: true
  level: "silent"  # No console spam, but still logs to JSON
```

### 3. Configure Email Notifications

#### Gmail Setup (Recommended)

1. Enable 2-factor authentication on your Gmail account
2. Create an app-specific password:
   - Go to https://myaccount.google.com/apppasswords
   - Select "Mail" and your device
   - Copy the generated 16-character password

3. Update `config.yaml`:

```yaml
notifications:
  enabled: true

  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    username: "your-email@gmail.com"
    password: "your-app-specific-password"
    from_address: "your-email@gmail.com"
    to_addresses:
      - "your-email@gmail.com"
```

#### Other Email Providers

**Outlook/Office 365:**
```yaml
smtp_server: "smtp-mail.outlook.com"
smtp_port: 587
```

**Yahoo:**
```yaml
smtp_server: "smtp.mail.yahoo.com"
smtp_port: 587
```

**Custom SMTP:**
```yaml
smtp_server: "mail.example.com"
smtp_port: 587
use_tls: true
```

### 4. Configure SMS Notifications (Optional)

SMS notifications require a Twilio account:

1. Sign up at https://www.twilio.com/
2. Get your Account SID and Auth Token from the dashboard
3. Purchase a phone number or use your trial number
4. Install Twilio SDK: `pip install twilio`

```yaml
notifications:
  sms:
    enabled: true
    account_sid: "your-account-sid"
    auth_token: "your-auth-token"
    from_number: "+1234567890"
    to_numbers:
      - "+1987654321"
```

### 5. Define Notification Rules

Rules determine when to send notifications. Each rule has:
- `name`: Descriptive name for the alert
- `enabled`: Whether this rule is active
- `cooldown_minutes`: Minimum time between notifications (prevents spam)
- `message`: Text to include in notification
- `conditions`: Event properties that must match

```yaml
notifications:
  rules:
    - name: "Cat at Food Bowl"
      enabled: true
      cooldown_minutes: 60
      message: "Your cat just visited the food bowl!"
      conditions:
        event_type: "ZONE_ENTER"
        zone_description: "food bowl"
        object_class_name: "cat"
```

## Advanced Pattern Matching

### Exact Match Conditions

```yaml
conditions:
  event_type: "LINE_CROSS"
  object_class_name: "cat"
  direction: "LTR"
```

### List Match (any of)

```yaml
conditions:
  object_class_name: ["cat", "dog"]
  zone_description: ["food bowl", "water bowl"]
```

### Comparison Operators

For numeric fields, use comparison operators:

```yaml
conditions:
  event_type: "ZONE_EXIT"
  zone_description: "food bowl"
  dwell_time:
    gte: 10  # Greater than or equal to 10 seconds
```

Available operators:
- `gt`: Greater than
- `gte`: Greater than or equal
- `lt`: Less than
- `lte`: Less than or equal
- `eq`: Equal to
- `ne`: Not equal to

### Example: Long Dwell Time Alert

Notify if a cat spends more than 20 seconds drinking:

```yaml
- name: "Cat Drinking Water"
  enabled: true
  cooldown_minutes: 120
  message: "Your cat spent significant time at the water bowl - staying hydrated!"
  conditions:
    event_type: "ZONE_EXIT"
    zone_description: "water bowl"
    object_class_name: "cat"
    dwell_time:
      gte: 20
```

### Example: Directional Movement

Notify when a cat crosses a specific line in a specific direction:

```yaml
- name: "Cat Entering Room"
  enabled: true
  cooldown_minutes: 30
  message: "Your cat just entered the bedroom"
  conditions:
    event_type: "LINE_CROSS"
    line_description: "bedroom doorway"
    object_class_name: "cat"
    direction: "LTR"
```

## Complete Example Configuration

Here's a full configuration for monitoring cats while away:

```yaml
# Minimal console output for long-running monitoring
console_output:
  enabled: true
  level: "silent"

# Define monitoring areas
zones:
  - x1_pct: 10
    y1_pct: 20
    x2_pct: 30
    y2_pct: 40
    description: "food bowl"
    allowed_classes: [15]

  - x1_pct: 40
    y1_pct: 20
    x2_pct: 60
    y2_pct: 40
    description: "water bowl"
    allowed_classes: [15]

# Notification configuration
notifications:
  enabled: true

  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    username: "yourname@gmail.com"
    password: "your-app-password"
    from_address: "yourname@gmail.com"
    to_addresses:
      - "yourname@gmail.com"

  sms:
    enabled: false  # Optional

  rules:
    # Alert when cat visits food bowl (max once per hour)
    - name: "Cat Eating"
      enabled: true
      cooldown_minutes: 60
      message: "Your cat visited the food bowl - good appetite!"
      conditions:
        event_type: "ZONE_ENTER"
        zone_description: "food bowl"
        object_class_name: "cat"

    # Alert when cat drinks water for significant time (max once per 2 hours)
    - name: "Cat Drinking"
      enabled: true
      cooldown_minutes: 120
      message: "Your cat spent time at the water bowl - staying hydrated!"
      conditions:
        event_type: "ZONE_EXIT"
        zone_description: "water bowl"
        object_class_name: "cat"
        dwell_time:
          gte: 5
```

## Running the System

### Short Test Run (1 hour)

```bash
python -m object_detection 1
```

### Long-term Monitoring (336 hours = 2 weeks)

```bash
python -m object_detection 336
```

### Run Until Manually Stopped

```bash
python -m object_detection 999999
```

Press `Ctrl+C` to stop at any time.

## Notification Email Format

When a rule triggers, you'll receive an email like this:

```
Subject: Object Detection Alert: Cat Eating

Your cat visited the food bowl - good appetite!

Event Details:
Time: 2025-12-22T14:35:47.123Z
Type: ZONE_ENTER
Object: cat
Zone: food bowl
```

## Notification SMS Format

SMS messages are concise (160 characters max):

```
Cat Eating: Your cat visited the food bowl - good appetite!
```

## Cooldown Periods

Cooldown prevents notification spam. Each rule tracks its last trigger time and won't fire again until the cooldown period expires.

**Example:** With `cooldown_minutes: 60`, you'll receive at most one notification per hour per rule, even if your cat visits the food bowl multiple times.

**Tips:**
- Use longer cooldowns (120-180 min) for frequent events
- Use shorter cooldowns (30-60 min) for rare events
- Critical alerts can have shorter cooldowns (15-30 min)

## Troubleshooting

### No Notifications Received

1. **Check logs:** System will log notification attempts
   ```
   INFO - Notification manager initialized with 2 rules
   INFO - Email notification sent: Cat at Food Bowl
   ```

2. **Verify email settings:**
   - Test SMTP credentials manually
   - Check spam folder
   - Verify app-specific password for Gmail

3. **Check rule conditions:**
   - Ensure zone descriptions match exactly
   - Verify object class names (use `detailed` console mode to see events)
   - Check cooldown hasn't suppressed notifications

### Email Authentication Errors

**Gmail "Username and Password not accepted":**
- Enable 2-factor authentication
- Create app-specific password (not your regular password)

**Connection refused:**
- Check firewall settings
- Verify SMTP server and port
- Ensure `use_tls: true` for most providers

### SMS Not Working

1. **Install Twilio SDK:**
   ```bash
   pip install twilio
   ```

2. **Verify credentials:** Check Account SID and Auth Token

3. **Check phone numbers:** Must include country code (e.g., `+1` for US)

## Privacy & Security

- **Email passwords:** Store in environment variables for production
- **JSONL files:** All events still logged regardless of notifications
- **Network activity:** Only occurs when sending notifications
- **Data privacy:** No event data sent to third parties (except chosen email/SMS providers)

## Environment Variables (Optional)

For better security, use environment variables instead of hardcoding credentials:

```bash
export EMAIL_PASSWORD="your-app-password"
export TWILIO_AUTH_TOKEN="your-auth-token"
```

Then reference in config:
```yaml
# This feature requires code modification - coming in future release
```

## Performance Impact

The notification system:
- ✅ Minimal CPU overhead (< 1% on most systems)
- ✅ No impact on detection FPS
- ✅ Runs asynchronously with detection
- ✅ Designed for long-term operation (days/weeks)

## What's Still Logged

Even in `silent` mode with notifications enabled:
- ✅ All events written to JSONL file
- ✅ System startup/shutdown messages
- ✅ Error messages and warnings
- ✅ Final statistics when stopping

## Support

For issues or questions:
- Check existing GitHub issues
- Review configuration examples
- Enable `detailed` console mode for debugging

Enjoy worry-free long-term monitoring! 🐱
