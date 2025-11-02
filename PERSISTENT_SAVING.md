# Persistent File Saving Feature

## Overview

The OpenDataTagger now includes robust persistent file saving that ensures your CSV tagging progress is saved even if the frontend disconnects or the browser is closed during processing.

## How It Works

### Automatic Saving Every 10 Rows
- The system automatically saves your tagged CSV and logs after every 10 rows processed
- Files are saved to the `media/` directory with predictable names:
  - `[original_filename]_tagged.csv` - Contains the original data plus new tagged columns
  - `[original_filename]_logs.csv` - Contains detailed logs of each tagging operation

### File Persistence Strategy
1. **Immediate Path Registration**: File paths are stored in Django cache as soon as tagging starts
2. **Batch Saving**: Progress is saved every 10 rows to balance performance and safety
3. **Recovery Mechanism**: If the frontend disconnects, files can be recovered using the original CSV name
4. **Error Handling**: Even if an error occurs, partial progress is saved

### Cache and Session Management
- File paths are stored in Django cache with 24-hour expiration (supports long-running jobs)
- Progress status is maintained in memory (`PROGRESS_STATUS` dictionary)
- Sessions are **NOT automatically cleaned** - manual cleanup only
- Conservative cleanup only removes completed/errored sessions older than 24 hours

## Usage

### Normal Operation
1. Upload your CSV file and define columns as usual
2. Start the tagging process
3. **Your progress is automatically saved every 10 rows**
4. If you close the browser, you can return and access the results

### Recovery After Disconnect
If your browser was closed during tagging:

1. Navigate back to the results page
2. The system will automatically look for saved files:
   - First checks the cache for stored file paths
   - Falls back to auto-generated filenames based on your original CSV
3. If files are found, you'll see your progress and can download the results

### Session Management
The system now uses **manual-only cleanup** to avoid interfering with long-running jobs.

#### List Active Sessions
```bash
cd AthensMT
python manage.py cleanup_sessions --list
```

#### Clean Up Old Completed Sessions (24+ hours old)
```bash
python manage.py cleanup_sessions
```

#### Clean Up Sessions Older Than Specific Hours
```bash
python manage.py cleanup_sessions --hours 48  # Clean sessions older than 48 hours
```

#### Dry Run (See What Would Be Cleaned)
```bash
python manage.py cleanup_sessions --dry-run
```

#### Force Cleanup All Sessions (USE WITH CAUTION)
```bash
python manage.py cleanup_sessions --force
```

## File Locations

All processed files are saved in the `media/` directory:
```
media/
├── your_data.csv                 # Original uploaded file
├── your_data_tagged.csv          # Tagged results (auto-saved every 10 rows)
├── your_data_logs.csv            # Processing logs (auto-saved every 10 rows)
└── your_data_config.csv          # Column definitions (if created)
```

## Benefits

1. **No Data Loss**: Even if the frontend crashes, your progress is preserved
2. **Resume Capability**: Can continue or review work from where you left off
3. **Monitoring**: Real-time progress tracking with persistent file status
4. **Error Recovery**: Partial results are saved even on unexpected errors
5. **Performance**: Batch saving every 10 rows optimizes disk I/O while ensuring safety

## Technical Details

### Backend Implementation
- **Threading**: Background processing continues independent of frontend
- **File I/O**: Uses pandas for reliable CSV operations
- **Caching**: Django cache system for session persistence
- **Error Handling**: Comprehensive exception handling with partial save recovery

### Frontend Integration
- **Progress Polling**: Real-time updates via AJAX calls
- **File Status**: Shows when files were last saved
- **Recovery UI**: Automatic detection and display of recovered files

## Troubleshooting

### Common Issues

**Q: I closed my browser during tagging. Where are my files?**
A: Navigate back to the results page. The system will automatically find your saved files based on your original CSV filename.

**Q: The tagging process seems stuck. Are my files safe?**
A: Yes! Files are saved every 10 rows. Check the `media/` directory for `*_tagged.csv` files.

**Q: How do I know if the background process is still running?**
A: Check the progress page - it shows real-time status and file save timestamps.

**Q: Can I run multiple tagging jobs simultaneously?**
A: Yes, each session has a unique key. Files are saved with the original CSV name as the base.

### File Recovery Commands
If you need to manually locate files:
```bash
# Find all tagged CSV files
find media/ -name "*_tagged.csv" -mtime -1

# Find recent log files
find media/ -name "*_logs.csv" -mtime -1

# Check file sizes to see progress
ls -lah media/*_tagged.csv
```

## Performance Considerations

- **Save Frequency**: Every 10 rows balances safety vs. performance
- **Memory Usage**: Progress data persists until manually cleaned (supports multi-hour jobs)
- **Disk Space**: Each save overwrites the previous file (no duplicates)
- **Cache Expiration**: File paths expire after 24 hours but files remain on disk
- **Long Jobs**: System designed to handle jobs that run for many hours without interference

## Long-Running Job Support

The system is now optimized for jobs that may take many hours:

1. **No Automatic Cleanup**: Sessions persist until manually cleaned
2. **Extended Cache**: File paths cached for 24 hours minimum
3. **Progress Tracking**: Last update timestamps prevent premature cleanup
4. **Conservative Cleanup**: Only cleans completed/errored sessions older than 24 hours
5. **File Recovery**: Files remain on disk even after session cleanup