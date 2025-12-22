# Email Notifications

Get email alerts when objects cross lines or enter/exit zones - perfect for long-term monitoring while you're away.

## Quick Start

### 1. Configure Email Settings

Edit `config.yaml`:

```yaml
notifications:
  enabled: true

  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    username: "your-email@gmail.com"
    password: "your-app-password"
    from_address: "your-email@gmail.com"
    to_addresses:
      - "your-email@gmail.com"
```

### 2. Tag Zones or Lines for Notifications

Add `notify_email: true` to any zone or line you want alerts for:

**Zone Example:**
```yaml
zones:
  - x1_pct: 10
    y1_pct: 20
    x2_pct: 30
    y2_pct: 40
    description: "food bowl"
    allowed_classes: [15]  # cats
    notify_email: true     # Get email when cat enters/exits
    cooldown_minutes: 60   # Max one email per hour
```

**Line Example:**
```yaml
lines:
  - type: vertical
    position_pct: 50
    description: "doorway"
    allowed_classes: [0]   # people
    notify_email: true     # Get email when line crossed
    cooldown_minutes: 30   # Max one email per 30 minutes
```

### 3. Run in Silent Mode (Optional)

For long-term monitoring with minimal console output:

```yaml
console_output:
  enabled: true
  level: "silent"
```

### 4. Start Monitoring

```bash
# Run for 2 weeks (336 hours)
python -m object_detection 336
```

## Email Configuration by Provider

### Gmail (Recommended)

1. Enable 2-factor authentication on your Google account
2. Create app-specific password:
   - Visit https://myaccount.google.com/apppasswords
   - Select "Mail" and your device
   - Copy the 16-character password

```yaml
email:
  enabled: true
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  use_tls: true
  username: "your-email@gmail.com"
  password: "abcd efgh ijkl mnop"  # Your 16-char app password
  from_address: "your-email@gmail.com"
  to_addresses:
    - "your-email@gmail.com"
```

### AWS SES

Great for high-volume notifications or if you're already using AWS:

1. Verify your sender email in AWS SES console
2. Generate SMTP credentials in SES console
3. Choose your region's SMTP endpoint

```yaml
email:
  enabled: true
  smtp_server: "email-smtp.us-east-1.amazonaws.com"  # Use your region
  smtp_port: 587
  use_tls: true
  username: "your-ses-smtp-username"  # From SES console
  password: "your-ses-smtp-password"  # From SES console
  from_address: "verified@yourdomain.com"  # Must be verified in SES
  to_addresses:
    - "your-email@example.com"
```

**AWS SES Regions:**
- `us-east-1`: email-smtp.us-east-1.amazonaws.com
- `us-west-2`: email-smtp.us-west-2.amazonaws.com
- `eu-west-1`: email-smtp.eu-west-1.amazonaws.com

### Outlook / Office 365

```yaml
email:
  enabled: true
  smtp_server: "smtp-mail.outlook.com"
  smtp_port: 587
  use_tls: true
  username: "your-email@outlook.com"
  password: "your-password"
  from_address: "your-email@outlook.com"
  to_addresses:
    - "your-email@outlook.com"
```

### Yahoo Mail

```yaml
email:
  enabled: true
  smtp_server: "smtp.mail.yahoo.com"
  smtp_port: 587
  use_tls: true
  username: "your-email@yahoo.com"
  password: "your-app-password"  # Generate at account.yahoo.com/account/security
  from_address: "your-email@yahoo.com"
  to_addresses:
    - "your-email@yahoo.com"
```

### Custom SMTP Server

```yaml
email:
  enabled: true
  smtp_server: "mail.yourdomain.com"
  smtp_port: 587
  use_tls: true
  username: "you@yourdomain.com"
  password: "your-password"
  from_address: "you@yourdomain.com"
  to_addresses:
    - "recipient@example.com"
```

## Notification Options

### Required

- `notify_email: true` - Enable notifications for this zone/line

### Optional

- `cooldown_minutes: 60` - Minimum time between notifications (default: 60)
- `message: "Custom text"` - Override default notification message

## What You'll Receive

### Zone Entry Email

```
Subject: Object Detection Alert: food bowl

A cat entered food bowl

Event Details:
Time: 2025-12-22T14:35:47.123Z
Type: ZONE_ENTER
Object: cat
Zone: food bowl
```

### Zone Exit Email

```
Subject: Object Detection Alert: water bowl

A cat exited water bowl (dwell: 12.4s)

Event Details:
Time: 2025-12-22T14:36:02.654Z
Type: ZONE_EXIT
Object: cat
Zone: water bowl
Dwell Time: 12.4s
```

### Line Cross Email

```
Subject: Object Detection Alert: doorway

A person crossed doorway (LTR)

Event Details:
Time: 2025-12-22T15:20:15.789Z
Type: LINE_CROSS
Object: person
Line: doorway
Direction: LTR
```

### Custom Message Email

If you specify a custom message:

```yaml
zones:
  - description: "food bowl"
    notify_email: true
    message: "Your cat is eating - good appetite!"
```

You'll receive:

```
Subject: Object Detection Alert: food bowl

Your cat is eating - good appetite!

Event Details:
Time: 2025-12-22T14:35:47.123Z
Type: ZONE_ENTER
Object: cat
Zone: food bowl
```

## Complete Example: Cat Monitoring

Monitor your cats' food and water consumption while traveling for 2 weeks:

```yaml
detection:
  model_file: "yolo11n.pt"
  track_classes: [15]  # cats only
  confidence_threshold: 0.25

console_output:
  enabled: true
  level: "silent"  # Minimal output

zones:
  # Food bowl monitoring
  - x1_pct: 10
    y1_pct: 20
    x2_pct: 30
    y2_pct: 40
    description: "food bowl"
    allowed_classes: [15]
    notify_email: true
    cooldown_minutes: 60
    message: "Your cat visited the food bowl!"

  # Water bowl monitoring
  - x1_pct: 40
    y1_pct: 20
    x2_pct: 60
    y2_pct: 40
    description: "water bowl"
    allowed_classes: [15]
    notify_email: true
    cooldown_minutes: 120
    message: "Your cat is staying hydrated!"

notifications:
  enabled: true
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    username: "your-email@gmail.com"
    password: "your-app-password"
    from_address: "your-email@gmail.com"
    to_addresses:
      - "your-email@gmail.com"
```

Run for 2 weeks:
```bash
python -m object_detection 336
```

## Cooldown Period

The cooldown prevents notification spam. Each zone/line tracks its last notification time independently.

**Example:** With `cooldown_minutes: 60`, you'll receive at most one email per hour for that specific zone, even if your cat visits it multiple times.

**Guidelines:**
- **Frequent events** (like food bowl visits): 60-120 minutes
- **Rare events** (like entering garage): 30-60 minutes
- **Critical alerts**: 15-30 minutes

## Multiple Recipients

Send notifications to multiple people:

```yaml
email:
  to_addresses:
    - "you@example.com"
    - "partner@example.com"
    - "petsitter@example.com"
```

## Troubleshooting

### No Emails Received

1. **Check logs** - System logs all notification attempts
2. **Verify email settings** - Test SMTP credentials
3. **Check spam folder**
4. **Verify zone/line configuration** - Ensure objects are actually triggering events
5. **Check cooldown** - May be suppressing notifications

### Gmail Authentication Errors

**"Username and Password not accepted":**
- Must use app-specific password, not your regular password
- Enable 2-factor authentication first
- Generate app password at https://myaccount.google.com/apppasswords

### AWS SES Issues

**"Email address not verified":**
- Verify sender email in SES console
- For production, verify domain instead of individual emails
- Check you're using correct region endpoint

**"Daily sending quota exceeded":**
- Check your SES sending limits in console
- Request limit increase if needed

### Connection Errors

**"Connection refused" or "Timeout":**
- Check firewall settings
- Verify SMTP server and port are correct
- Ensure `use_tls: true` for most providers
- Try port 465 with SSL instead of 587 with TLS (rare)

### Still Logging Events Without Notifications

**This is normal!** Events are ALWAYS logged to JSON files regardless of notification settings. Notifications are an optional alert mechanism on top of the logging.

## Privacy & Security

- **Credentials**: Consider using environment variables in production instead of hardcoding passwords
- **Data**: No event data is sent to third parties except your chosen email provider
- **Network**: SMTP connections use TLS encryption
- **Logs**: All events still saved to local JSONL files

## Performance

- **CPU Impact**: < 1% overhead
- **FPS Impact**: None - notifications run asynchronously
- **Network**: Only sends emails when events occur (respects cooldown)
- **Long-term**: Designed for weeks of continuous operation

## What's Still Logged

Even in silent mode:
- ✅ All events written to JSONL file
- ✅ System startup/shutdown messages
- ✅ Errors and warnings
- ✅ Final statistics

Email notifications are an alert layer on top of the comprehensive logging system.
