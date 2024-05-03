
print("""<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Downloader</title>
    <style>
        element.style {
            background-color: rgb(53, 54, 58);
        }

        body {
            font-family: Roboto, 'Segoe UI', Tahoma, sans-serif;
            font-size: 81.25%;
        }

        body {
            background: #35363A;
            margin: 0;
        }
        form{
            margin-top:15rem;
            margin-left:32rem
        }

        label {
            color:aquamarine
        }
        p{
            margin-left:3rem;
            margin-top:2rem;
            color: azure;
            font-size: 3rem;
            font-weight: bolder;

        }
         a{
           
            color :lightgrey;
            text-decoration: none;
            font-weight:bold;
        }

        
    </style>
</head>

<body>
    <p>Roy Downloader</p>""")

import re

string = "1. kitchen Sink - Twenty One Pilots 26. Stop The World ( I Wanna Get Off With You ) - Arctic Monkeys 2.27 - Passenger 27. Breezeblocks - Al J 3. Forest - Twenty One Pilots 28. The Run And Go - Twenty One Pilots 4. Migraine - Twenty One Pilots 29. Honey - Kehlani 5. Hunger - Florence and the Machine 30. Fake You Out - Twenty One Pilots 6. What The Water Gave Me - Florence and The Machine 31. When The Day Met The Night - Panic At The Disco 7. Kissland - The Weeknd 32. Do You See What I'm Seeing ? - Panic ! At The Disco 8. Ultralife - Oh Wonder 33. Death Of A Bachelor - Panic ! At The Disco 9. Lose It - Oh Wonder 34. Castle - Habey 10. Fade - Jakwob ft . Maiday 35. How Will I Know - Whitney Houston 11. The Only Exception - Paramore 36. Somewhere Only We know - Keane 12. Wonderful Wonderful - The Killers 37. Coming Home - The Kaiser Chief 13. Latch - Disclosure ft . Sam Smith 38 , Dancing In The Moonlight - Toploader 14. Deaderush- Alt 39. The Less I Know The Better - Tame Impala 15. Whistle For The Choir - The Fratellis 10. I Follow Rivers - Lykke Li ( Magician's Remix ) 16. The Calling - The Killers 41. Walls - Kings Of Leon 17. The Longest Wave - The Red Hot Chilli Peppers 12. 505 - Aretic Monkeys 18. Sun - Two Door Cinema Club 43. Reckless Serenade - Arctic Monkeys 19. Non Believers - La Rocca 44. The End - Kings of Leon 20. Big God - Florence and the Machine 45. When I Was Your Man - Bruno Mars 21. Too Much Is Never Enough - Florence and the Machine 46. Home - Gabrielle Aplin 12. Dreams - Fleetwood Mac 17. Beautiful War -Kings Of Leon 23. Tell Me Lies - Fleetwood Mac 18. Madness -- Muse 24. I Wanna Be Your Lover - Prince 19. Sorry - Nothing But Thieves 25. I Just Don't Know What To Do With MyselfThe White Stripes 50. Fix You - Coldplay "

lsen= re.findall ("\D+",string)
for sen in lsen:
    m=sen.replace(". ","https://youtubetomp3music.com/en16/download?url=")
    k=m.replace(" ","+")
    print(f"<a href={k}>{sen}</a><br>")
print("""</body>

</html>""")
