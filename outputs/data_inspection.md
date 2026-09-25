# Data inspection

- Rows: 50
- Columns: 16

## Column profile

| Column | Inferred role | Missing | Unique | Sample values |
|---|---|---:|---:|---|
| People, B2B AI Startups, Marketing Tech | empty | 100% | 0 |  |
| First Name | identifier | 0% | 46 | Gary; Kevin; Ryan; Steven |
| Last Name | identifier | 0% | 48 | Vaynerchuk; O'Leary; Reynolds; Bartlett |
| Full Name | identifier | 0% | 50 | Gary Vaynerchuk; Kevin O'Leary; Ryan Reynolds; Steven Bartlett |
| Job Title | free-form semantic text | 0% | 42 | CEO; Investor, Strategic Advisor; Co-Founder; Investor |
| Company | categorical / short text | 0% | 47 | Vayner3; Tax Hive; Group Effort Initiative; Stan |
| City | categorical / short text | 24% | 28 | West Palm Beach; New York; Thai Nguyen; Jacksonville |
| State or Province | categorical / short text | 28% | 18 | New York; Florida; California; British Columbia |
| Country | categorical | 0% | 12 | United States; United Kingdom; Vietnam; Singapore |
| LinkedIn Profile | identifier | 0% | 50 | https://www.linkedin.com/in/garyvaynerchuk/; https://www.linkedin.com/in/kevinolearytv/; https://www.linkedin.com/in/vancityreynolds/; https://www.linkedin.com/in/stevenbartlett-123/ |
| Enrich person | categorical / short text | 88% | 6 | Kevin O'Leary; Steven Bartlett; Carolina Martins; Reid Hoffman |
| Connections | numerical | 88% | 4 | 427.0; 29029.0; 500.0; 4324.0 |
| Headline | free-form semantic text | 88% | 6 | Chairman, O’Leary Ventures and Beanstox; Founder: Steven.com The Creator Economy Company, home to FlightStory & FlightCast.; Mulher mais seguida do LinkedIn na América Latina \| Founder-Led Growth \| LinkedIn para Executivos \| Palestrante; Co-Founder, LinkedIn, Manas AI & Inflection AI. Founding Team, PayPal.  Author of Superagency.  Podcaster of Possible and Masters of Scale.   |
| Summary | free-form semantic text | 88% | 6 | Kevin O'Leary's success story starts where most entrepreneurs begin: with a big idea and zero cash. From his basement, he launched SoftKey Software Products. As sales took off, Kev; Steven Bartlett is a 33-year-old British entrepreneur, investor in more than 100 companies, speaker, author, and media owner. He is the founder of Steven.com - the creator media co; O LinkedIn é uma sala de reunião antecipada.<br><br>70% dos decisores afirmam que estão mais propensos a fazer negócios com líderes que produzem conteúdo relevante na rede.<br>Aind; My current priority is investing in and building with AI to benefit humanity.  Recently, I've co-founded Inflection.AI and ManasAI.co.    I am active in all facets of the consumer  |
| Jobs Count | numerical | 88% | 5 | 12.0; 25.0; 5.0; 33.0 |
| Summarize LinkedIn profile | free-form semantic text | 88% | 6 | Kevin O'Leary is a successful entrepreneur and media personality known for his investments.; Unique aspects about Steven Bartlett:; 1. Most followed woman on LinkedIn in Latin America, with over 3 million followers. 2. Founder of...; 1. Co-founded LinkedIn, revolutionizing professional networking, now with over 1 billion users. 2... |

## Candidate semantic views

| Rank | Field | Coverage | Diversity |
|---:|---|---:|---:|
| 1 | Job Title | 100% | 84% |
| 2 | Headline | 12% | 12% |
| 3 | Summary | 12% | 12% |
| 4 | Summarize LinkedIn profile | 12% | 12% |

Numerical fields are kept as source attributes and are not embedded. The all-null first column is preserved in the output but excluded from modeling.
