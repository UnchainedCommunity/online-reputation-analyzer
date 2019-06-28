import twint

# TODO recover tweets about the active topics
from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('cron', hour=0)
def scheduled_job():
    print('This job is run every weekday at 00pm.')
    # Configure
    c = twint.Config()
    c.Search = "#djezzy"
    c.Format = "Tweet id: {id} | Tweet: {tweet}"

    # Run)
    twint.run.Search(c)

    # TODO Sentiment analysis using textblob, textblob-fr, textblob-ar, textblob-dz


print('This job is run every weekday at 00pm.')
# Configure
c = twint.Config()
c.Username = "noneprivacy"
c.Search = "#djezzy"
c.Format = "Tweet id: {str(id)} | Tweet: {tweet}"

# Run)
twint.run.Search(c)
sched.start()