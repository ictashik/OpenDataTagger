from django.core.management.base import BaseCommand
from tagger_app.utils import cleanup_abandoned_sessions, PROGRESS_STATUS
import time

class Command(BaseCommand):
    help = 'Clean up old completed/errored tagging sessions (manual only - no automatic cleanup)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--hours',
            type=int,
            default=24,
            help='Only clean sessions older than this many hours (default: 24)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be cleaned without actually cleaning',
        )
        parser.add_argument(
            '--list',
            action='store_true',
            help='List all current sessions and their status',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force cleanup of all sessions regardless of status (USE WITH CAUTION)',
        )

    def handle(self, *args, **options):
        hours = options['hours']
        dry_run = options['dry_run']
        list_sessions = options['list']
        force = options['force']
        
        if list_sessions:
            self._list_sessions()
            return
        
        if force:
            self.stdout.write(
                self.style.WARNING(
                    'WARNING: Force cleanup will remove ALL sessions including running ones!'
                )
            )
            confirm = input('Type "yes" to continue: ')
            if confirm != 'yes':
                self.stdout.write('Cleanup cancelled.')
                return
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Cleaning sessions older than {hours} hours (dry-run: {dry_run})'
            )
        )

        if dry_run:
            self._dry_run_cleanup(hours)
        else:
            if force:
                self._force_cleanup()
            else:
                cleanup_abandoned_sessions(hours)
                self.stdout.write(
                    self.style.SUCCESS('Session cleanup completed')
                )

    def _list_sessions(self):
        """List all current sessions and their status."""
        if not PROGRESS_STATUS:
            self.stdout.write('No active sessions found.')
            return
        
        self.stdout.write(f'Found {len(PROGRESS_STATUS)} active sessions:')
        current_time = time.time()
        
        for session_key, data in PROGRESS_STATUS.items():
            start_time = data.get("start_time", current_time)
            last_update = data.get("last_update", start_time)
            age_hours = (current_time - start_time) / 3600
            idle_minutes = (current_time - last_update) / 60
            
            self.stdout.write(f'  Session: {session_key[:8]}...')
            self.stdout.write(f'    Status: {data.get("status", "unknown")}')
            self.stdout.write(f'    Progress: {data.get("done", 0)}/{data.get("total", 0)} rows')
            self.stdout.write(f'    Age: {age_hours:.1f} hours')
            self.stdout.write(f'    Last update: {idle_minutes:.1f} minutes ago')
            self.stdout.write('')

    def _dry_run_cleanup(self, hours):
        """Show what would be cleaned without actually cleaning."""
        current_time = time.time()
        cleanup_threshold = hours * 3600
        would_cleanup = []
        
        for session_key, data in PROGRESS_STATUS.items():
            session_start_time = data.get("start_time", current_time)
            session_age = current_time - session_start_time
            status = data.get("status", "")
            
            should_cleanup = (
                session_age > cleanup_threshold and (
                    status == "finished" or 
                    status.startswith("error") or
                    status == "Completed" or
                    (session_age > cleanup_threshold and "Processing row" not in status)
                )
            )
            
            if should_cleanup:
                would_cleanup.append((session_key, data))
        
        if would_cleanup:
            self.stdout.write(f'Would clean up {len(would_cleanup)} sessions:')
            for session_key, data in would_cleanup:
                age_hours = (current_time - data.get("start_time", current_time)) / 3600
                self.stdout.write(f'  {session_key[:8]}... - {data.get("status")} - {age_hours:.1f}h old')
        else:
            self.stdout.write('No sessions would be cleaned up.')

    def _force_cleanup(self):
        """Force cleanup of all sessions (dangerous)."""
        count = len(PROGRESS_STATUS)
        PROGRESS_STATUS.clear()
        self.stdout.write(
            self.style.WARNING(f'Force cleaned {count} sessions')
        )