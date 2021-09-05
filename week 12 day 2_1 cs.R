#1. Build h2o::automl(). For this task:
#• prepare data using tk_augment_timeseries_signature()
#• set stopping metric to “RMSE”
#• set exclude_algos = c("DRF", "GBM","GLM",'XGBoost')


library(tidymodels)
library(modeltime)
library(tidyverse)
library(lubridate)
library(timetk)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)
library(lime)
library(inspectdf)
library(caret)
library(glue)
library(scorecard)
library(mice)
library(plotly)
library(recipes) 
library(purrr) 
library(graphics) 
library(Hmisc) 
library(inspectdf)
library(tidymodels)
library(modeltime)
library(timetk)

path <- dirname(getSourceEditorContext()$path)
setwd(path)

raw <- fread("temperatures.csv")

colnames(raw) <- c('Date','Count')

raw %>% glimpse()

raw %>% inspect_na()

names(raw)<- raw %>% names() %>% gsub(" ","_",.)

raw[raw$Date=="7/20/1982", "Count"] <- "0.2"
raw[raw$Date=="7/21/1982", "Count"] <- "0.8"
raw[raw$Date=="7/14/1984", "Count"] <- "0.1"

raw$Date <- raw$Date %>% as.factor() %>% as.Date(.,"%m/%d/%Y") 
raw$Count <- raw$Count %>% as.numeric() 

raw %>% inspect_na()
raw %>% glimpse()

#h2o.init()  

#raw <-raw %>% tk_augment_timeseries_signature(Date) %>% select(Count)

#raw %>%inspect_na()

#raw$diff %>% unique()

#raw$diff %>% table()

#raw[is.na(raw$diff),]$diff <- 86400

#raw %>% dim()

#raw %>%glimpse()

#raw$month.lbl <- raw$month.lbl %>% as.character()

#raw$wday.lbl <- raw$wday.lbl %>% as.character()

h2o.init()  

raw_tk <- raw %>% tk_augment_timeseries_signature()


df <- raw_tk %>%
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute,-second,-am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character,as_factor)


train_h2o <- df %>% filter(Date < 1986) %>% as.h2o()
test_h2o <- df %>% filter(Date >= 1986) %>% as.h2o()

target <- "Count" 
features <- df %>% select(-Count) %>% names()

model_h2o <- h2o.automl(
  x = features, y = target, 
  training_frame = train_h2o, 
  validation_frame = test_h2o,
  leaderboard_frame = test_h2o,
  stopping_metric = "RMSE",
  seed = 123, nfolds = 10,
  exclude_algos = c("DRF", "GBM","GLM",'XGBoost'),
  max_runtime_secs = 480) 

model_h2o@leaderboard %>% as.data.frame() 
h2o_leader <- model_h2o@leader

#2. Build modeltime::arima_reg(). For this task set engine to “auto_arima”
#3. 3. Forecast temperatures for next year with model which has lower RMSE.

splits <- initial_time_split(raw, prop = 0.9)

model_fit_arima_no_boost <- arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  fit(Count ~ Date,  data = training(splits))


model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Count ~ Date, data = training(splits))


model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(Count ~ Date, data = training(splits))

recipe_spec <- recipe(Count ~ Date, data = training(splits)) %>%
  step_date(date, features = "month", ordinal = FALSE) %>%
  step_mutate(date_num = as.numeric(date)) %>%
  step_normalize(date_num) %>%
  step_rm(date)

models_tbl <- modeltime_table(
  model_fit_arima_no_boost,
  model_fit_ets,
  model_fit_prophet
)

models_tbl


calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))
calibration_tbl

calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = TRUE
  )

refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = raw)

refit_tbl %>%
  modeltime_forecast(h = "3 years", actual_data = raw) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = TRUE
  )
