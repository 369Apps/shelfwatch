# I built a shelf stock monitor with Roboflow in an afternoon

I run a barbershop. We have a small retail shelf: pomades, beard oils,
shampoos. Nobody checks it until a customer asks for something that is not
there. That is lost money hiding in plain sight.

So I built ShelfWatch. You snap a photo of the shelf with your phone. It
detects every product on the shelf, counts them, and tells you what is
running low. Seconds, not a clipboard walk.

The whole thing runs on Roboflow. I used a retail shelf detection model from
Roboflow Universe, called it through their serverless inference API with
their Python SDK, and wrapped it in about 100 lines of my own code for the
counting and the restock report. No training. No GPU. No server to babysit.

What surprised me: the hardest part was not the computer vision. It was
deciding what "low stock" means for a shelf this small. Three units felt
right for us. That is a business rule, not a model parameter, and getting it
wrong would make the whole tool useless no matter how good the detections
are.

Next I want to train a custom model on our actual SKUs so it names products
instead of counting shapes, and push the low stock alerts to our shop
WhatsApp instead of a terminal. The vision part is solved. The last mile is
where the money is.

Code and setup: one script, one free Roboflow API key, one photo.
